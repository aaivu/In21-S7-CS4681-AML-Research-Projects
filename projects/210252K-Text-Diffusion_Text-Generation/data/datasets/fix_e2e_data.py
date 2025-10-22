import os

def fix_e2e_encoding(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in ['src1_train.txt', 'src1_valid.txt', 'src1_test.txt']:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"Skipping {filename} - not found")
            continue
            
        # Read with latin-1 to handle the corrupted encoding
        with open(input_path, 'r', encoding='latin-1') as f:
            content = f.read()
        
        # Fix common encoding issues
        content = content.replace('Ã‚Â£', '£')
        content = content.replace('CafÃƒÂ©', 'Café')
        content = content.replace('CafÃƒ Â©', 'Café')
        
        # Write with proper UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f'Fixed {filename}')

# Run this
fix_e2e_encoding('../datasets/e2e_data', '../datasets/e2e_data_fixed')