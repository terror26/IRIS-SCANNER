import numpy as np
import matplotlib.pyplot as plt
import abel

CartImage = abel.tools.analytical.sample_image(501)[201:-200, 201:-200]

CartImage = cv2.imread('01_L.bmp',0)
PolarImage, r_grid, theta_grid = abel.tools.polar.reproject_image_into_polar(CartImage)

fig, axs = plt.subplots(1,2, figsize=(7,3.5))
axs[0].imshow(CartImage , aspect='auto', origin='lower')
axs[1].imshow(PolarImage, aspect='auto', origin='lower', 
              extent=(np.min(theta_grid), np.max(theta_grid), np.min(r_grid), np.max(r_grid)))

axs[0].set_title('Cartesian')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].set_title('Polar')
axs[1].set_xlabel('Theta')
axs[1].set_ylabel('r')

plt.tight_layout()
plt.show()