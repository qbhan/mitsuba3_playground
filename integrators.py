# from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import pyexr
import matplotlib.pyplot as plt
import numpy as np
import gc

from typing import *

mi.set_variant('cuda_ad_rgb')


def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)

class Simple(mi.SamplingIntegrator):
    '''
    Simple Path Tracer with
    1. BSDF Sampling
    2. Simple Russian Roulette
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get("max_depth")
        self.rr_depth = props.get("rr_depth")

    def sample(self, 
               scene: mi.Scene, 
               sampler: mi.Sampler, 
               ray_: mi.RayDifferential3f, 
               medium: mi.Medium = None, 
               active: bool = True):
        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(ray_)
        depth = mi.UInt32(0)
        f = mi.Spectrum(1.)
        L = mi.Spectrum(0.)

        prev_si = dr.zeros(mi.SurfaceInteraction3f)


        # --------------------- Configure loop state ----------------------

        loop = mi.Loop(name="Path Tracing", state=lambda: (
            sampler, ray, depth, f, L, active, prev_si))

        loop.set_max_iterations(self.max_depth)


        # -------------------------- Start Loop ----------------------------

        while loop(active):


            # ------ Compute detailed record of the current intersection ------

            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

            
            # ---------------------- Direct emission ----------------------

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            Le = f * ds.emitter.eval(si)


           # ------------------------- BSDF sampling -------------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()
            bsdf: mi.BSDF = si.bsdf(ray)

            # BSDF sampling
            bsdf_smaple, bsdf_val = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            # Update loop variables
            ray = si.spawn_ray(si.to_world(bsdf_smaple.wo))
            L = (L + Le)
            f *= bsdf_val

            prev_si = dr.detach(si, True)

            # ------------ Stopping criterion (Russian Roulette) ------------

            # Don't run another iteration if the throughput has reached zero
            active_next &= dr.neq(dr.max(f), 0)

            # Simple Russian roulette stopping probability
            rr_prop = dr.maximum(f.x, dr.maximum(f.y, f.z))
            rr_prop[depth < self.rr_depth] = 1.
            f *= dr.rcp(rr_prop)
            active_next &= (sampler.next_1d() < rr_prop)

            active = active_next
            depth += 1
        return (L, dr.neq(depth, 0), [])


class MyPathIntegrator(mi.SamplingIntegrator):
    '''
    Path Integrator including the followings:
    1. Multiple Importance Sampling
    2. Russian Roulette
    3. Next Event Estimation
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get("max_depth")
        self.rr_depth = props.get("rr_depth")


    def sample(self, 
               scene: mi.Scene, 
               sampler: mi.Sampler, 
               ray_: mi.RayDifferential3f, 
               medium: mi.Medium = None, 
               active: bool = True):
        bsdf_ctx = mi.BSDFContext()


        # --------------------- Configure loop state ----------------------

        ray = mi.Ray3f(ray_)
        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        depth = mi.UInt32(0)                            # Depth of current vertex
        L = mi.Spectrum(0.0)                            # Radiance accumulator
        β = mi.Spectrum(1.0)                            # Path throughput weight
        η = mi.Float(1)                                 # Index of refraction
        mis_em = mi.Float(1)                            # Emitter MIS weight
        active = mi.Bool(active)                        # Active SIMD lanes

        loop = mi.Loop(name="Path Tracing", state=lambda: (
            sampler, ray, prev_si, depth, L, β, η, mis_em, active))

        loop.set_max_iterations(self.max_depth)

        while loop(active):


            # ------ Compute detailed record of the current intersection ------
            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

    
            # ---------------------- Direct emission ----------------------

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            Le = β * mis_em * ds.emitter.eval(si)


            # ----------------------- Emitter sampling -----------------------
            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # Get the BSDF
            bsdf: mi.BSDF = si.bsdf(ray)

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si,
                                                            wo, active_em)
            mis_direct = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
            Lr_dir = β * mis_direct * bsdf_value_em * em_weight


            # ------------------------- BSDF sampling -------------------------
            
            # Perform detached BSDF sampling.
            bsdf_sample, bsdf_val = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            bsdf_sample_delta = mi.has_flag(bsdf_sample.sampled_type,
                                        mi.BSDFFlags.Delta)

            # Update loop variables
            η     *= bsdf_sample.eta
            β     *= bsdf_val
            L = L + Le + Lr_dir

            prev_si = dr.detach(si, True)


            # ------------ Stopping criterion (Russian Roulette) ------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue


            # ------------------ Intersect next surface -------------------

            ray_next = si.spawn_ray(si.to_world(bsdf_sample.wo))
            pi_next = scene.ray_intersect_preliminary(ray_next,
                                                      active=active_next)

            # Compute a detached intersection record for the next vertex
            si_next = pi_next.compute_surface_interaction(ray_next)


            # ---------- Compute MIS weight for the next vertex -----------

            ds = mi.DirectionSample3f(scene, si=si_next, ref=si)

            # Probability of sampling 'si_next' using emitter sampling
            # (set to zero if the BSDF doesn't have any smooth components)
            pdf_em = scene.pdf_emitter_direction(
                ref=si, ds=ds, active=~bsdf_sample_delta
            )

            mis_em = mis_weight(bsdf_sample.pdf, pdf_em)

            # Provide ray/interaction to the next iteration
            pi   = pi_next
            ray  = ray_next

            active = active_next
            depth += 1
        return (L, dr.neq(depth, 0), [])


class MyPathAOVIntegrator(mi.SamplingIntegrator):
    '''
    Path AOV Integrator including the followings:
    1. Multiple Importance Sampling
    2. Russian Roulette
    3. Next Event Estimation
    4. Extracting Geometric features
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get("max_depth")
        self.rr_depth = props.get("rr_depth")
        self.aov_list = props.get("aovs").split(',')
        self.aov_support = ['normal:3', 'depth:1', 'albedo:3']
        

    def aovs(self):
        # parse aov_list and check all aovs are supported
        self.aovs = []
        for aov in self.aov_list:
            if aov not in self.aov_support:
                assert NotImplementedError('AOV {} is currently not supported'.format(aov))
            aov_name, aov_ch = aov.split(':')[0], aov.split(':')[1]
            for i in range(int(aov_ch)):
                self.aovs.append(aov_name + '.{}'.format(str(i+1)))

        print(self.aovs)
        return self.aovs


    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True
            ) -> mi.TensorXf:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )
            # Generate a set of rays starting at the sensor
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
            # Launch the Monte Carlo sampling process in primal mode
            L, valid, aovs = self.sample(
                scene=scene,
                sampler=sampler,
                ray_=ray,
                medium=None,
                active=mi.Bool(True)
            )

            # Prepare an ImageBlock as specified by the film
            block = sensor.film().create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)
            # Accumulate into the image block
            alpha = dr.select(valid, mi.Float(1), mi.Float(0))
            if mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special):
                aovs = sensor.film().prepare_sample(L * weight, ray.wavelengths,
                                                    block.channel_count(), alpha=alpha)
                block.put(pos, aovs)
                del aovs
            else:
                rgb = L * weight
                block_input = [rgb[0], rgb[1], rgb[2], mi.Float(1.0)] + aovs
                block.put(pos, block_input)

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid, alpha
            gc.collect()

            # Perform the weight division and return an image tensor
            sensor.film().put_block(block)
            self.primal_image = sensor.film().develop()

            return self.primal_image



    def prepare(self,
                sensor: mi.Sensor,
                seed: int = 0,
                spp: int = 0,
                aovs: list = []):
        """
        Given a sensor and a desired number of samples per pixel, this function
        computes the necessary number of Monte Carlo samples and then suitably
        seeds the sampler underlying the sensor.
        Returns the created sampler and the final number of samples per pixel
        (which may differ from the requested amount depending on the type of
        ``Sampler`` being used)
        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor to render the scene from a different viewpoint.
        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator during the primal rendering step. It is crucial that you
            specify different seeds (e.g., an increasing sequence) if subsequent
            calls should produce statistically independent images (e.g. to
            de-correlate gradient-based optimization steps).
        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            primal rendering step. The value provided within the original scene
            specification takes precedence if ``spp=0``.
        """

        film = sensor.film()
        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = dr.prod(film_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)
        a = film.prepare(aovs)
        # print(a)

        return sampler, spp

    
    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        reparam: Callable[[mi.Ray3f, mi.UInt32, mi.Bool],
                          Tuple[mi.Vector3f, mi.Float]] = None
        ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        Sample a 2D grid of primary rays for a given sensor
        Returns a tuple containing
        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray
        When a reparameterization function is provided via the 'reparam'
        argument, it will be applied to the returned image-space position (i.e.
        the sample positions will be moving). The other two return values
        remain detached.
        """

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += mi.Vector2i(film.crop_offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        reparam_det = 1.0

        if reparam is not None:
            if rfilter.is_box_filter():
                raise Exception(
                    "ADIntegrator detected the potential for image-space "
                    "motion due to differentiable shape or camera pose "
                    "parameters. This is, however, incompatible with the box "
                    "reconstruction filter that is currently used. Please "
                    "specify a smooth reconstruction filter in your scene "
                    "description (e.g. 'gaussian', which is actually the "
                    "default)")

            # This is less serious, so let's just warn once
            if not film.sample_border() and self.sample_border_warning:
                self.sample_border_warning = True

                mi.Log(mi.LogLevel.Warn,
                    "ADIntegrator detected the potential for image-space "
                    "motion due to differentiable shape or camera pose "
                    "parameters. To correctly account for shapes entering "
                    "or leaving the viewport, it is recommended that you set "
                    "the film's 'sample_border' parameter to True.")

            with dr.resume_grad():
                # Reparameterize the camera ray
                reparam_d, reparam_det = reparam(ray=dr.detach(ray),
                                                 depth=mi.UInt32(0))

                # TODO better understand why this is necessary
                # Reparameterize the camera ray to handle camera translations
                if dr.grad_enabled(ray.o):
                    reparam_d, _ = reparam(ray=ray, depth=mi.UInt32(0))

                # Create a fake interaction along the sampled ray and use it to
                # recompute the position with derivative tracking
                it = dr.zeros(mi.Interaction3f)
                it.p = ray.o + reparam_d
                ds, _ = sensor.sample_direction(it, aperture_sample)

                # Return a reparameterized image position
                pos_f = ds.uv + film.crop_offset()

        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

        return ray, weight, splatting_pos, reparam_det


    def sample(self, scene: mi.Scene,
               sampler: mi.Sampler, 
               ray_: mi.RayDifferential3f, 
               medium: mi.Medium = None, 
               active: bool = True
               ):
        bsdf_ctx = mi.BSDFContext()


        # --------------------- Configure loop state ----------------------

        ray = mi.Ray3f(ray_)
        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        depth = mi.UInt32(0)                            # Depth of current vertex
        L = mi.Spectrum(0.0)                            # Radiance accumulator
        β = mi.Spectrum(1.0)                            # Path throughput weight
        η = mi.Float(1)                                 # Index of refraction
        mis_em = mi.Float(1)                            # Emitter MIS weight
        active = mi.Bool(active)                        # Active SIMD lanes
        
        aov = [mi.Float(0.0) for i in range(len(self.aovs))]
        first_non_specular = mi.Bool(False)
        depth_diff = mi.Float(0.0)

        loop = mi.Loop(name="Path Tracing", state=lambda: (
            sampler, ray, prev_si, depth, L, β, η, mis_em, active, aov, first_non_specular, depth_diff))

        loop.set_max_iterations(self.max_depth)

        while loop(active):


            # ------ Compute detailed record of the current intersection ------

            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

            # accumulate depth
            depth_diff += dr.select(dr.eq(depth, 0) & si.is_valid(), si.t, mi.Float(0.0))

    
            # ---------------------- Direct emission ----------------------

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            Le = β * mis_em * ds.emitter.eval(si)


            # ----------------------- Emitter sampling -----------------------
            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # Get the BSDF
            bsdf: mi.BSDF = si.bsdf(ray)

            # check bsdf specular
            found_specular = si.is_valid() & (mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta) | mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta1D))

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si,
                                                            wo, active_em)
            mis_direct = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
            Lr_dir = β * mis_direct * bsdf_value_em * em_weight


            # ------------------------- BSDF sampling -------------------------
            
            # Perform detached BSDF sampling.
            bsdf_sample, bsdf_val = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            bsdf_sample_delta = mi.has_flag(bsdf_sample.sampled_type,
                                        mi.BSDFFlags.Delta)

            # Update loop variables
            η     *= bsdf_sample.eta
            β     *= bsdf_val
            L = L + Le + Lr_dir

            prev_si = dr.detach(si, True)

            # -------------- Store AOVs (Geometric Features) -------------

            # record normal, depth, and albedo of first bounce
            if 'normal:3' in self.aov_list:
                aov[self.aovs.index('normal.1')] += dr.select(('normal.1' in self.aovs) & dr.eq(depth, 0) & si.is_valid(), si.sh_frame.n.x, mi.Float(0.0))
                aov[self.aovs.index('normal.2')] += dr.select(('normal.2' in self.aovs) & dr.eq(depth, 0) & si.is_valid(), si.sh_frame.n.y, mi.Float(0.0))
                aov[self.aovs.index('normal.3')] += dr.select(('normal.3' in self.aovs) & dr.eq(depth, 0) & si.is_valid(), si.sh_frame.n.z, mi.Float(0.0))
            if 'depth:1' in self.aov_list:
                aov[self.aovs.index('depth.1')] += dr.select(('depth.1' in self.aovs) & dr.eq(depth, 0) & si.is_valid(), si.t, mi.Float(0.0))
            albedo = bsdf.eval_diffuse_reflectance(si, active)
            if 'albedo:3' in self.aov_list:
                aov[self.aovs.index('albedo.1')] += dr.select(('albedo.1' in self.aovs) & dr.eq(depth, 0) & si.is_valid(), albedo[0], mi.Float(0.0))
                aov[self.aovs.index('albedo.2')] += dr.select(('albedo.2' in self.aovs) & dr.eq(depth, 0) & si.is_valid(), albedo[1], mi.Float(0.0))
                aov[self.aovs.index('albedo.3')] += dr.select(('albedo.3' in self.aovs) & dr.eq(depth, 0) & si.is_valid(), albedo[2], mi.Float(0.0))

            # record normal, depth, and albedo of first non-specular bounce
            if 'normal_diff:3' in self.aov_list:
                aov[self.aovs.index('normal_diff.1')] += dr.select(('normal_diff.1' in self.aovs) & dr.eq(first_non_specular, False) & dr.eq(found_specular, False) & si.is_valid(), si.sh_frame.n.x, mi.Float(0.0))
                aov[self.aovs.index('normal_diff.2')] += dr.select(dr.eq(('normal_diff.2' in self.aovs) & first_non_specular, False) & dr.eq(found_specular, False) & si.is_valid(), si.sh_frame.n.y, mi.Float(0.0))
                aov[self.aovs.index('normal_diff.3')] += dr.select(dr.eq(('normal_diff.3' in self.aovs) & first_non_specular, False) & dr.eq(found_specular, False) & si.is_valid(), si.sh_frame.n.z, mi.Float(0.0))
            if 'depth_diff:1' in self.aov_list:
                aov[self.aovs.index('depth_diff.1')] += dr.select(dr.eq(('depth_diff.1' in self.aovs) & first_non_specular, False) & dr.eq(found_specular, False) & si.is_valid(), depth_diff, mi.Float(0.0))
            if 'albedo_diff:3' in self.aov_list:
                aov[self.aovs.index('albedo_diff.1')] += dr.select(('albedo_diff.1' in self.aovs) & dr.eq(first_non_specular, False) & dr.eq(found_specular, False) & si.is_valid(), albedo[0], mi.Float(0.0))
                aov[self.aovs.index('albedo_diff.2')] += dr.select(('albedo_diff.2' in self.aovs) & dr.eq(first_non_specular, False) & dr.eq(found_specular, False) & si.is_valid(), albedo[1], mi.Float(0.0))
                aov[self.aovs.index('albedo_diff.3')] += dr.select(('albedo_diff.3' in self.aovs) & dr.eq(first_non_specular, False) & dr.eq(found_specular, False) & si.is_valid(), albedo[2], mi.Float(0.0))

            # toggle it off
            first_non_specular = dr.select(dr.eq(first_non_specular, False) & dr.eq(found_specular, True), False, True)


            # ------------ Stopping criterion (Russian Roulette) ------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue


            # ------------------ Intersect next surface -------------------

            ray_next = si.spawn_ray(si.to_world(bsdf_sample.wo))
            pi_next = scene.ray_intersect_preliminary(ray_next,
                                                      active=active_next)

            # Compute a detached intersection record for the next vertex
            si_next = pi_next.compute_surface_interaction(ray_next)


            # ---------- Compute MIS weight for the next vertex -----------

            ds = mi.DirectionSample3f(scene, si=si_next, ref=si)

            # Probability of sampling 'si_next' using emitter sampling
            # (set to zero if the BSDF doesn't have any smooth components)
            pdf_em = scene.pdf_emitter_direction(
                ref=si, ds=ds, active=~bsdf_sample_delta
            )

            mis_em = mis_weight(bsdf_sample.pdf, pdf_em)

            # Provide ray/interaction to the next iteration
            pi   = pi_next
            ray  = ray_next

            active = active_next
            depth += 1
        return (L, dr.neq(depth, 0), aov)


if __name__ == "__main__":
    # mi.set_log_level(mi.LogLevel.Info)
    mi.register_integrator("MyPathAOV", lambda props: MyPathAOVIntegrator(props))
    mi.register_integrator("MyPath", lambda props: MyPathIntegrator(props))
    mi.register_integrator("simple", lambda props: Simple(props))

    # specify aovs
    aovs = [
        'normal:3',
        'depth:1',
        'albedo:3',
        'normal_diff:3',
        'depth_diff:1',
        'albedo_diff:3'
    ]

    my_path = mi.load_dict({
        'type': 'MyPathAOV',
        'max_depth': 6,
        'rr_depth': 5,
        'aovs': ','.join(aovs)
    })
    scene = mi.load_file("bathroom/scene.xml")
    import time
    start = time.time()
    image = mi.render(scene, spp=64, integrator=my_path)
    np_image = np.array(image)
    end = time.time()
    if np_image.shape[-1] == 3:
        mi.util.write_bitmap("my_first_render.png", np_image)
    else:
        from utils import tonemap, save_aovs
        import matplotlib.pyplot as plt
        # rgb, normal, depth, albedo = np_image[:, :, :3], np_image[:, :, 3:6], np_image[:, :, 6], np_image[:, :, 7:10]
        # plt.imsave("rgb.png", tonemap(rgb))
        # normal = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
        # plt.imsave("result/normal.png", normal)
        # plt.imsave("result/depth.png", depth, cmap='gray', vmin=np.min(depth), vmax=np.max(depth))
        # plt.imsave('result/albedo.png', albedo)
        save_aovs(np_image, aovs)
    print('rendering time:', end-start)