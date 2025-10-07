import torchattacks
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
# If inputs were normalized, then
# atk.set_normalization_used(mean=[...], std=[...])
adv_images = atk(images, labels)