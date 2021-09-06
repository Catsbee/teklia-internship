import json


file = open("text_classification_example.ipynb", "r")
p = file.readlines()
print(p)
print((type(p[0])))
st = ''.join(p)
liste = json.loads(st)

print(type(liste))
tab = liste['cells']
co = ''
for i in tab:
    if i['cell_type'] == 'code':
        co += ''.join(i['source'])
        co += '\n'
print(co)

prog = open("output.py", "w")
prog.writelines(co)
print(prog)
prog.close()

file.close()

