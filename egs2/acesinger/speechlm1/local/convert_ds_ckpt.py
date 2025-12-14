import torch
import sys

'''
Sample Scripts
python local/convert_ds_ckpt.py exp/speechlm_opuslm_v1_1.7B_anneal_ext_phone_finetune_cot_svs_trynl/checkpoint_62/62/mp_rank_00_model_states.pt exp/speechlm_opuslm_v1_1.7B_anneal_ext_phone_finetune_cot_svs_trynl/finetune_62epoch.pth
'''
in_file, out_file = sys.argv[1], sys.argv[2]

state_dict = torch.load(in_file, map_location='cpu', weights_only=False)
new_dict = {"model": state_dict["module"]}
# new_dict = state_dict['module']
torch.save(new_dict, out_file)