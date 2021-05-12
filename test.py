# def l2genotype(l, num_node):
#     PRIMITIVES = [
#     'sdf',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect', # identity
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5',
#     'none'
#     ]
#     genotype = 'Genotype(normal=['
#     for i in range(num_node):
#         genotype += '[(\'%s\', %d), (\'%s\', %d)],'%(PRIMITIVES[l[i*4]], l[i*4+1], PRIMITIVES[l[i*4+2]], l[i*4+3])
#     genotype += '], normal_concat=range(2, %d), reduce=['%(num_node + 2)
#     for i in range(num_node, num_node*2):
#         genotype += '[(\'%s\', %d), (\'%s\', %d)],'%(PRIMITIVES[l[i*4]], l[i*4+1], PRIMITIVES[l[i*4+2]], l[i*4+3])
#     genotype += '], reduce_concat=range(2, %d))'%(num_node + 2)
#     print(genotype)
# l2genotype([1,0,2,1,3,1,4,2,5,2,6,3,7,3,8,4,1,0,2,1,3,1,4,2,5,2,6,3,7,3,8,4], 4)

# import numpy as np
# from geneticalgorithm import geneticalgorithm as ga

# def f(X):
#     return -np.sum(X)


# varbound=np.array([[0,10]]*3)

# model=ga(function=f,dimension=3,variable_type='int',variable_boundaries=varbound)

# model.run()
# print(model.report)
# print(model.output_dict)

# def ga_opt(num_node):
#     bound = []
#     for i in range(num_node):
#         bound += [[0,6], [0,i+1], [0,6], [0,i+1]]
#     bound *= 2
#     print(bound)
# ga_opt(4)

import GPy
import numpy as np
def bayes_opt(num_node):
    dim      = 32      # input's dim
    num_init = 4
    val_batches = 30
    score_epochs = 1
    max_eval = 50
    # xs       = generate_random_arch.random_gen(num_init, num_node)
    # ys       = np.zeros(num_init)
    # for i in range(num_init):
    #     ys[i] = train_val(score_epochs, gt.from_str(l2genotype(xs[i], num_node)), val_batches)
    # print(xs, ys)
    xs =   [[0,0,1,1,2,0,6,1,1,0,3,1,1,3,6,1,3,0,5,0,5,2,1,2,6,3,1,0,4,3,1,2],
            [0,0,2,0,0,1,6,1,6,1,2,0,0,4,1,3,4,0,4,0,6,0,5,2,3,1,2,1,6,1,0,3],
            [6,1,0,1,4,2,2,2,5,0,0,0,4,3,3,0,4,0,0,1,6,2,6,0,2,0,0,1,5,1,0,3],
            [4,1,2,1,1,1,3,2,4,1,1,1,0,0,5,2,6,0,4,1,5,2,6,0,0,3,1,0,5,2,4,3]]*10
    # xs = (np.array(xs)-3)/2
    ys = [0.56948933,0.59156493, 0.64, 0.61128812]*10
    # ys = (np.array(ys)-0.6)*5
    print(xs, ys)
    for cnt in range(max_eval):
        y1 = ys.reshape(len(ys), 1)
        gp_m1 = GPy.models.GPRegression(xs, y1, GPy.kern.RBF(input_dim = dim, ARD = True))
        gp_m1.kern.variance = np.var(y1)
        print(np.std(xs, axis=0))
        # gp_m1.kern.lengthscale = np.std(xs, axis=0)
        gp_m1.likelihood.variance = 1e-4 * np.var(y1)
        gp_m1.optimize()

        def lcb(x):
            py1, ps2_1 = gp_m1.predict(x.reshape(1, dim)-3)
            ps_1       = np.sqrt(ps2_1)
            lcb1       = py1 - 3 * ps_1
            return lcb1[0, 0]
        
        new_x = ga_opt(num_node, lcb)
        new_y = train_val(score_epochs, gt.from_str(l2genotype(new_x, num_node)), val_batches)
        
        # ga_result = ga.run(1000, mdenas.rand_arch(50), population = 50, fix_fun = lcb)

        # idx = np.random.permutation(len(ga_result))[:4]
        # new_x = ga_result[idx]
        # new_y = np.zeros(4)
        # for i in range(4):
        #     new_y[i] = mdenas.predict_acc(xs[i], thread = 4)
        
        py1, ps2_1 = gp_m1.predict(new_x.reshape(1, dim))
        
        print(cnt)
        print('-'*60)

        print(py1, ps2_1, new_y)
        ys = np.append(ys, new_y)
        xs = np.concatenate((xs, new_x.reshape(1, dim)-3), axis=0)

bayes_opt(4)
