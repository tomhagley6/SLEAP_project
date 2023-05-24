NODES = {'nose':0, 'earleft':1, 'earright':2,
             'neck':3, 'bodyupper':4, 'bodylower':5,
               'tailbase':6}

def get_node_numbers(node_name):

    nodes = NODES
    
    try:
        node_name = node_name.lower()

    except AttributeError as e:
        print(e)
        print("Please input node_name as a string")

    try:

        return nodes[node_name]
    
    except KeyError as e:
        print(e)
        print(f"node_name: {node_name} not a node")    

        return None
    