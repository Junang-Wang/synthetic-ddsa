from skimage import io
import napari
from vsystem.visuals import plot_tiff, plot_tiffs

files = ["./vessels/volumes/Lnet_d20_dr12_epsilon9_iter13_SD500_v1_t0_512x512x280.tiff", "./vessels/volumes/Lnet_d20_dr15_epsilon9_iter13_SD500_v1_t0_512x512x280.tiff","./vessels/volumes/Lnet_d20_dr20_epsilon9_iter13_SD500_v1_t0_512x512x280.tiff"]

noFluid_files = ["./vessels/volumes/Lnet_d20_dr12_epsilon9_iter13_SD500_v1_t0_512x512x280_nofluid.tiff", "./vessels/volumes/Lnet_d20_dr15_epsilon9_iter13_SD500_v1_t0_512x512x280_nofluid.tiff","./vessels/volumes/Lnet_d20_dr20_epsilon9_iter13_SD500_v1_t0_512x512x280_nofluid.tiff", "./vessels/volumes/Lnet_d35_dr20_epsilon10_iter8_SD500_v1_t0_512x512x280_nofluid.tiff"]

# plot_tiffs(files)

plot_tiffs(noFluid_files)


