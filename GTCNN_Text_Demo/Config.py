from Util import *

#======================= Settings =======================#
Setting = Empty()

Setting.taoOFR = 0
Setting.kappaOFR = 0.9
Setting.kappa0 = 0.7
Setting.tao0 = 20
Setting.epsi0 = 1

Setting.batch_size = 25
Setting.SweepTime = 4
Setting.Iter = 200
Setting.Burin = 0.6 * Setting.Iter
Setting.Collection = Setting.Iter - Setting.Burin

Setting.K = [200]
Setting.T = len(Setting.K)

#======================= SuperParams =======================#
SuperParams = Empty()

SuperParams.gamma0 = 0.1  # r
SuperParams.c0 = 0.1
SuperParams.a0 = 0.1  # p
SuperParams.b0 = 0.1
SuperParams.e0 = 0.1  # c
SuperParams.f0 = 0.1
SuperParams.eta = 0.05  # Phi