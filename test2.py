import numpy as np
import math
from collections import namedtuple
import sys
from graphviz import Digraph

# import visualize
# import genotypes as gt
def l2graph(l, num_node):
    G_m = np.zeros([num_node+2, num_node+2])
    C_m = np.zeros([num_node+2, num_node+2])
    C_m[0, 0] = 0.15
    C_m[1,1] = 0.25
    for i in range(num_node):
        G_m[i+2, l[i*4+1]] -= 2**l[i*4]
        G_m[l[i*4+1], i+2] -= 2**l[i*4]
        G_m[i+2, i+2] += 2**l[i*4]
        G_m[l[i*4+1], l[i*4+1]] += 2**l[i*4]

        G_m[i+2, l[i*4+3]] -= 2**l[i*4+2]
        G_m[l[i*4+3], i+2] -= 2**l[i*4+2]
        G_m[i+2, i+2] += 2**l[i*4+2]
        G_m[l[i*4+3], l[i*4+3]] += 2**l[i*4+2]

        C_m[i+2, i+2]  = 1/math.sqrt(2**l[i*4]+2**l[i*4+2])
        
    t_m = np.dot(C_m, G_m)
    A_m = np.dot(t_m, C_m)
    eigvals, eigvecs = np.linalg.eig(A_m)
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    temp = eigvecs[:, 1:3]
    # print(G_m)
    # print(C_m)
    # print(A_m)
    # print(eigvals)
    return abs(np.reshape(temp.transpose(), [2*(num_node+2)]))

def gene_dist(l, num_node):
    d1 = l2graph(l[:4*num_node], num_node)
    d2 = l2graph(l[4*num_node:], num_node)
    # print(d1, d2)
    return np.append(d1, d2)

def l2genotype(l, num_node):
    PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
    ]
    genotype = 'Genotype(normal=['
    for i in range(num_node):
        genotype += '[(\'%s\', %d), (\'%s\', %d)],'%(PRIMITIVES[l[i*4]], l[i*4+1], PRIMITIVES[l[i*4+2]], l[i*4+3])
    genotype += '], normal_concat=range(2, %d), reduce=['%(num_node + 2)
    for i in range(num_node, num_node*2):
        genotype += '[(\'%s\', %d), (\'%s\', %d)],'%(PRIMITIVES[l[i*4]], l[i*4+1], PRIMITIVES[l[i*4+2]], l[i*4+3])
    genotype += '], reduce_concat=range(2, %d))'%(num_node + 2)
    return genotype
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def plot(genotype, file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        # 'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        # 'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor='lightblue')

    for i, edges in enumerate(genotype):
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', )

    g.render(file_path, view=False)

x1 = [5,0,5,1,2,1,1,0,0,3,1,0,2,3,6,3, 6,1,1,1,6,1,6,2,6,3,6,3,3,3,0,4]
x2 = [5,0,4,1,2,1,5,0,0,3,1,1,2,3,4,3, 6,1,1,1,6,1,6,2,6,3,6,3,3,3,0,4]
x3 = [6,1,6,0,0,1,3,0,6,2,1,3,3,4,0,2, 1,1,0,0,6,2,4,0,6,1,6,0,0,3,2,3]

# d1 = l2graph(x1[:16], 4)
# d2 = l2graph(x2[:16], 4)
# d3 = l2graph(x3[:16], 4)
# print(np.linalg.norm(d1-d2, ord=2))
# print(np.linalg.norm(d1-d3, ord=2))

g1 = eval(l2genotype(x1, 4))
g2 = eval(l2genotype(x2, 4))
g3 = eval(l2genotype(x3, 4))
plot(g1.normal, "normal1", 'g1')
plot(g2.normal, "normal2", 'g2')
plot(g3.normal, "normal3", 'g3')