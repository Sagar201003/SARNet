import json
import os

def fix_notebook():
    nb_path = r'c:\Users\shiva\OneDrive\Desktop\SARNet\Notebooks\sarnetv2_FINAL.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            
            # 1. Config fixes
            if 'class Config:' in source:
                source = source.replace('BATCH_SIZE = 4      # safe for P100 16GB at 256×256', 
                                        'BATCH_SIZE = 8      # Increased for 2x T4 GPUs (4 per GPU)')
                source = source.replace('LAMBDA_ID   = 5.0   # identity loss weight', 
                                        'LAMBDA_ID   = 0.0   # identity loss weight (disabled due to channel mismatch)')
                
            # 2. Model Instantiate (Add DataParallel)
            if '# ── INSTANTIATE' in source:
                dp_code = """
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    G_AB = nn.DataParallel(G_AB)
    G_BA = nn.DataParallel(G_BA)
    D_A = nn.DataParallel(D_A)
    D_B = nn.DataParallel(D_B)
"""
                if 'nn.DataParallel' not in source:
                    source = source.replace('D_B  = PatchGANDiscriminator(cfg.OPT_CHANNELS, cfg.NDF).to(device)\n',
                                            'D_B  = PatchGANDiscriminator(cfg.OPT_CHANNELS, cfg.NDF).to(device)\n' + dp_code)
                    
            # 3. Checkpoint Saving (Save space!)
            if 'def save_checkpoint' in source:
                new_save_code = """def save_checkpoint(epoch, is_best=False, extra=None):
    \"\"\"
    Saves last checkpoint and best checkpoint.
    Prevents Kaggle from running out of disk space!
    \"\"\"
    def get_state_dict(model):
        return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
    ckpt = {
        'epoch'  : epoch,
        'G_AB'   : get_state_dict(G_AB),
        'G_BA'   : get_state_dict(G_BA),
        'D_A'    : get_state_dict(D_A),
        'D_B'    : get_state_dict(D_B),
        'opt_G'  : opt_G.state_dict(),
        'opt_D_A': opt_D_A.state_dict(),
        'opt_D_B': opt_D_B.state_dict(),
        'history': dict(history),
    }
    if extra:
        ckpt.update(extra)

    # ONLY SAVE LAST MODEL INSTEAD OF EVERY EPOCH (Saves space!)
    epoch_path = os.path.join(cfg.CHECKPOINT_DIR, 'last_model.pth')
    torch.save(ckpt, epoch_path)

    if is_best:
        best_path = os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth')
        torch.save(ckpt, best_path)
        print(f'  ★ BEST model saved  (epoch {epoch})')

def load_best_model():
    best_path = os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        if isinstance(G_AB, nn.DataParallel):
            G_AB.module.load_state_dict(ckpt['G_AB'])
            G_BA.module.load_state_dict(ckpt['G_BA'])
        else:
            G_AB.load_state_dict(ckpt['G_AB'])
            G_BA.load_state_dict(ckpt['G_BA'])
        print(f'Best model loaded from epoch {ckpt["epoch"]}')
        return ckpt
    else:
        print('No best_model.pth found. Using current weights.')
        return None"""
                # Replace the old functions
                import re
                source = re.sub(r'def save_checkpoint.*?return None', new_save_code, source, flags=re.DOTALL)
                
            # 4. Train one epoch identity loss bug
            if 'def train_one_epoch' in source:
                old_id_code = """            # id_A   = G_BA(real_A)       # SAR identity
            # id_B   = G_AB(real_B)       # Optical identity
            # AFTER
            id_A   = G_AB(real_A)       # SAR identity
            id_B   = G_BA(real_B)       # Optical identity

            loss_adv  = criterion_GAN(D_B(fake_B), rl) + criterion_GAN(D_A(fake_A), rl)
            loss_cyc  = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * cfg.LAMBDA_CYC
            loss_id   = (criterion_id(id_A, real_A)     + criterion_id(id_B, real_B))     * cfg.LAMBDA_ID"""
                
                new_id_code = """            loss_adv  = criterion_GAN(D_B(fake_B), rl) + criterion_GAN(D_A(fake_A), rl)
            loss_cyc  = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * cfg.LAMBDA_CYC
            
            # Identity loss disabled because of 1ch vs 3ch mismatch between SAR and Optical
            loss_id   = torch.tensor(0.0).to(device)"""
                
                source = source.replace(old_id_code, new_id_code)

            # Re-split by lines to put back into list of strings correctly for jupyter format
            lines = source.splitlines(True)
            cell['source'] = lines

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook modified successfully.")

if __name__ == "__main__":
    fix_notebook()
