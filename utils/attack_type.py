from attacks import *
import numpy as np

attackers = {
    'fgsm': lambda predict, eps, nb_iter, eps_iter: FGSM(predict, eps=eps),
    'mi_fgsm': lambda predict, loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: MomentumIterativeAttack(predict, loss_fn, eps, nb_iter, eps_iter,\
                                                        decay_factor=1., ord=np.inf),
    'pgd': lambda predict, eps, nb_iter, eps_iter: PGDAttack(predict, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True,\
                                                        ord=np.inf, l1_sparsity=None),
}