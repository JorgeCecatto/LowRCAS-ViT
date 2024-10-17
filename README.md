# LowRCAS-ViT: 
This project aims to add adapters to the proposed architecture in 

https://github.com/Tianfang-Zhang/CAS-ViT

Thank you for the public repository and your incredible contributions

## Run

```bash
docker build --tag image_name .
```
```bash
docker run --it image_name docker_name 
```
It is a base image, based on running on a cpu, to use a gpu consider changing the image from python to pytorch and ensure you have a version of the nvidia tool kit installed

After that, continue execution and add the necessary files following https://github.com/Tianfang-Zhang/CAS-ViT

If you have problems with the cv2 package, run the following commands
```bash
pip uninstall opencv-python
```
and 
```bash
pip install opencv-python-headless
```

# Project in progress 🚀
