#EXT := sep_reset

default: EDSR_baseline_x3

# EDSR baseline model (x2)
EDSR_baseline_x2:
	python main.py --model EDSR --scale 2 --save EDSR_baseline_x2 --reset $(EXT)

# EDSR baseline model (x3) - from EDSR baseline model (x2)
EDSR_baseline_x3:
	python main.py --model EDSR --scale 3 --save EDSR_baseline_x3 --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt

# EDSR baseline model (x4) - from EDSR baseline model (x2)
EDSR_baseline_x4:
	python main.py --model EDSR --scale 4 --save EDSR_baseline_x4 --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt

# EDSR in the paper (x2)
EDSR_x2:
	python main.py --model EDSR --scale 2 --save EDSR_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
EDSR_x3:
	python main.py --model EDSR --scale 3 --save EDSR_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_best.pt

# EDSR in the paper (x4) - from EDSR (x2)
EDSR_x4:
	python main.py --model EDSR --scale 4 --save EDSR_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_best.pt

# MDSR baseline model
MDSR_baseline:
	python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
MDSR:
	python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
Test-EDSR_baseline_x4_test_Set5:
	python main.py --data_test Set5 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble

Test-EDSR_baseline_x4_test_Set14:
	python main.py --data_test Set14 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble

Test-EDSR_baseline_x4_test_B100:
	python main.py --data_test B100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble

Test-EDSR_baseline_x4_Urban100:
	python main.py --data_test Urban100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble

Test-EDSR_baseline_x4_test_DIV2K:
	python main.py --data_test DIV2K --ext img --n_val 100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble

Test-EDSR_x4_test_Set5:
	python main.py --data_test Set5 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble

Test-EDSR_x4_test_Set14:
	python main.py --data_test Set14 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble

Test-EDSR_x4_test_B100:
	python main.py --data_test B100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble

Test-EDSR_x4_test_Urban100:
	python main.py --data_test Urban100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble

Test-EDSR_x4_test_DIV2K:
	python main.py --data_test DIV2K --ext img --n_val 100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble

# Test your own images
Test-EDSR_baseline_x4:
	python main.py --data_test Demo --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --save_results

# Advanced - Test with JPEG images
Test-MDSR_baseline_jpeg: 
	python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train ../experiment/model/MDSR_baseline_jpeg.pt --test_only --save_results

# Advanced - Training with adversarial loss
Train-EDSR_GAN:
	python main.py --template GAN --scale 4 --save EDSR_GAN --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train ../experiment/model/EDSR_baseline_x4.pt
