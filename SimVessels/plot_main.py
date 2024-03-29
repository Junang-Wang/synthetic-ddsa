from skimage import io
import napari
from vsystem.visuals import plot_tiff, plot_tiffs

files = ["./vessels/volumes/Lnet_d20_dr12_epsilon9_iter13_SD500_v1_t600_280x512x512.tiff", "./vessels/volumes/Lnet_d20_dr15_epsilon9_iter13_SD500_v1_t600_280x512x512.tiff","./vessels/volumes/Lnet_d20_dr20_epsilon9_iter13_SD500_v1_t600_280x512x512.tiff"]

noFluid_files = ["./vessels/volumes/Lnet_d20_dr12_epsilon9_iter13_SD500_v1_t600_280x512x512_nofluid.tiff", "./vessels/volumes/Lnet_d20_dr15_epsilon9_iter13_SD500_v1_t600_280x512x512_nofluid.tiff","./vessels/volumes/Lnet_d20_dr20_epsilon9_iter13_SD500_v1_t600_280x512x512_nofluid.tiff"]

plot_tiffs(noFluid_files+files)

# plot_tiffs(noFluid_files)


