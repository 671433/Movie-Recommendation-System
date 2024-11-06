import re

# open requirements.txt
with open('requirements.txt', encoding='utf-16') as file:
    lines = file.readlines()

# replace c\\
new_lines = []
for line in lines:
    match = re.match(r'([a-zA-Z0-9_.-]+)(\s*@\s*file://.*)', line)
    if match:
        new_lines.append(f"{match.group(1)}\n")
    else:
        new_lines.append(line)

# write
with open('requirements.txt', 'w') as file:
    file.writelines(new_lines)

print("Updated requirements.txt ")
