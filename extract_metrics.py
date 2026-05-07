import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_metrics(file_path, keywords):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'stream':
                        text = "".join(output.get('text', []))
                        for line in text.split('\n'):
                            if any(k in line for k in keywords):
                                print(f"[{file_path}] {line.strip()}")
                    elif output.get('output_type') in ['execute_result', 'display_data']:
                        text = "".join(output.get('data', {}).get('text/plain', []))
                        for line in text.split('\n'):
                            if any(k in line for k in keywords):
                                print(f"[{file_path}] {line.strip()}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print("--- CycleGAN Metrics ---")
get_metrics(r'C:\Users\shiva\OneDrive\Desktop\SARNet\Notebooks\sarnetv2_FINAL.ipynb', ['PSNR', 'SSIM', '★', 'Epoch 54'])

print("\n--- Classifier Metrics ---")
get_metrics(r'C:\Users\shiva\OneDrive\Desktop\SARNet\Notebooks\sar_classifier.ipynb', ['Val Acc', 'Accuracy', 'Epoch 15'])
