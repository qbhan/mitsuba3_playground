import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')
scene = mi.load_file("scenes/cbox.xml")
image = mi.render(scene, spp=256, res=1024)
print(image.shape)

# plt.axis("off")
# # plt.plot(image ** (1.0 / 2.2)); # approximate sRGB tonemapping
# plt.imsave('savefig_default.png', image ** (1.0 / 2.2))
mi.util.write_bitmap("savefig_default.exr", image)