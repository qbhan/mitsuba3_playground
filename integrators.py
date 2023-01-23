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

    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray_: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True):
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

    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray_: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True):
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

            ray_next = si.spawn_ray(si.to_world(bsdf_sample_delta.wo))
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


if __name__ == "__main__":
    # mi.set_log_level(2)
    # dr.set_device(1)
    mi.register_integrator("MyPath", lambda props: MyPathIntegrator(props))
    mi.register_integrator("simple", lambda props: Simple(props))
    my_path = mi.load_dict({
        'type': 'simple',
        'max_depth': 6,
        'rr_depth': 5
    })
    scene = mi.load_file("bathroom/scene.xml")
    import time
    start = time.time()
    image = mi.render(scene, spp=64, integrator=my_path)
    print(image.shape, type(image))
    mi.util.write_bitmap("my_first_render.png", image)
    end = time.time()
    print('rendering time:', end-start)