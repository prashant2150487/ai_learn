writing_list=['sen','ben', 'john']
writing_list.sort()

for index,item in enumerate(writing_list):
    row=f"{index+1}.{item.capitalize()}"
    print(row)

