import os
for root, dirs, filenames in os.walk('.'):
    for f in filenames:
        if f.endswith('.py'):
            fp = os.path.join(root, f)
            with open(fp, 'rb') as fh:
                data = fh.read()
            if b'\x00' in data:
                print(f'Cleaning: {fp}')
                with open(fp, 'wb') as fh:
                    fh.write(data.replace(b'\x00', b''))
print('Done')