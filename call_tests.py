from test_eval_model import *




#generate_images_with_generator(16,"EKSPERYMENTY_GAN/generatory/gen80.pth","EKS_GAM","802")
#print_model()




#interpolation_experiment(10,"EKSPERYMENT_INTERPOLACJA/gen80.pth", "EKSPERYMENT_INTERPOLACJA/vae99.pth","EKSPERYMENT_INTERPOLACJA",2)

#compare_generated_images_with_oryginals(64,"EKSPERYMENT_WARUNKOWE/vae99.pth", "EKSPERYMENT_WARUNKOWE/gen99.pth","EKSPERYMENT_WARUNKOWE")


#calculate_MSE_for_fake_masks(1,1000,"EKSPERYMENT_WARUNKOWE/vae99.pth", "EKSPERYMENT_WARUNKOWE/gen99.pth","EKSPERYMENTY_INCEPTION_SCORE")

#calculate_SSIM_for_fake_images(1000,"EKSPERYMENT_WARUNKOWE/vae99.pth", "EKSPERYMENT_WARUNKOWE/gen99.pth","EKSPERYMENTY_INCEPTION_SCORE")


#generate_whole_set()

#measure_architecture_times("EKSPERYMENTY_INCEPTION_SCORE")
test_triangle_masks_on_models(8,"EKSPERYMENT_WARUNKOWE/vae99.pth", "EKSPERYMENT_WARUNKOWE/gen99.pth","EKSPERYMENTY_INCEPTION_SCORE")

#compare_generated_images_with_oryginals(8,"EKSPERYMENT_WARUNKOWE/vae99.pth", "EKSPERYMENT_WARUNKOWE/gen99.pth","EKSPERYMENTY_INCEPTION_SCORE/fin")

#final_animation_test(9, "EKSPERYMENT_WARUNKOWE/vae99.pth","EKSPERYMENTY_INCEPTION_SCORE")