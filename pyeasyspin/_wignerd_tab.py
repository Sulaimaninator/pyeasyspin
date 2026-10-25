"""Explicit Wigner little-d formulas for J<=4, generated from
EasySpin wignerd.m (littled_explicit). Keys (J, linidx); linidx is MATLAB column-major, 1-based."""
import numpy as np

TAB = {}

def _d_1_1(beta):
    return np.cos(beta/2)
TAB[(0.5, 1)] = _d_1_1

def _d_1_2(beta):
    return np.sin(beta/2)
TAB[(0.5, 2)] = _d_1_2

def _d_1_3(beta):
    return -np.sin(beta/2)
TAB[(0.5, 3)] = _d_1_3

def _d_1_4(beta):
    return np.cos(beta/2)
TAB[(0.5, 4)] = _d_1_4

def _d_2_1(beta):
    return (1+np.cos(beta))/2
TAB[(1.0, 1)] = _d_2_1

def _d_2_2(beta):
    return np.sin(beta)/np.sqrt(2)
TAB[(1.0, 2)] = _d_2_2

def _d_2_3(beta):
    return (1-np.cos(beta))/2
TAB[(1.0, 3)] = _d_2_3

def _d_2_4(beta):
    return -np.sin(beta)/np.sqrt(2)
TAB[(1.0, 4)] = _d_2_4

def _d_2_5(beta):
    return np.cos(beta)
TAB[(1.0, 5)] = _d_2_5

def _d_2_6(beta):
    return np.sin(beta)/np.sqrt(2)
TAB[(1.0, 6)] = _d_2_6

def _d_2_7(beta):
    return (1-np.cos(beta))/2
TAB[(1.0, 7)] = _d_2_7

def _d_2_8(beta):
    return -np.sin(beta)/np.sqrt(2)
TAB[(1.0, 8)] = _d_2_8

def _d_2_9(beta):
    return (1+np.cos(beta))/2
TAB[(1.0, 9)] = _d_2_9

def _d_3_1(beta):
    return np.cos(beta/2)**3
TAB[(1.5, 1)] = _d_3_1

def _d_3_2(beta):
    return np.sqrt(3)*np.sin(beta/2)*np.cos(beta/2)**2
TAB[(1.5, 2)] = _d_3_2

def _d_3_3(beta):
    return np.sqrt(3)*np.sin(beta/2)**2*np.cos(beta/2)
TAB[(1.5, 3)] = _d_3_3

def _d_3_4(beta):
    return np.sin(beta/2)**3
TAB[(1.5, 4)] = _d_3_4

def _d_3_5(beta):
    return -np.sqrt(3)*np.sin(beta/2)*np.cos(beta/2)**2
TAB[(1.5, 5)] = _d_3_5

def _d_3_6(beta):
    return np.cos(beta/2)*(3*np.cos(beta/2)**2-2)
TAB[(1.5, 6)] = _d_3_6

def _d_3_7(beta):
    return -np.sin(beta/2)*(3*np.sin(beta/2)**2-2)
TAB[(1.5, 7)] = _d_3_7

def _d_3_8(beta):
    return np.sqrt(3)*np.sin(beta/2)**2*np.cos(beta/2)
TAB[(1.5, 8)] = _d_3_8

def _d_3_9(beta):
    return np.sqrt(3)*np.sin(beta/2)**2*np.cos(beta/2)
TAB[(1.5, 9)] = _d_3_9

def _d_3_10(beta):
    return np.sin(beta/2)*(3*np.sin(beta/2)**2-2)
TAB[(1.5, 10)] = _d_3_10

def _d_3_11(beta):
    return np.cos(beta/2)*(3*np.cos(beta/2)**2-2)
TAB[(1.5, 11)] = _d_3_11

def _d_3_12(beta):
    return np.sqrt(3)*np.sin(beta/2)*np.cos(beta/2)**2
TAB[(1.5, 12)] = _d_3_12

def _d_3_13(beta):
    return -np.sin(beta/2)**3
TAB[(1.5, 13)] = _d_3_13

def _d_3_14(beta):
    return np.sqrt(3)*np.sin(beta/2)**2*np.cos(beta/2)
TAB[(1.5, 14)] = _d_3_14

def _d_3_15(beta):
    return -np.sqrt(3)*np.sin(beta/2)*np.cos(beta/2)**2
TAB[(1.5, 15)] = _d_3_15

def _d_3_16(beta):
    return np.cos(beta/2)**3
TAB[(1.5, 16)] = _d_3_16

def _d_4_1(beta):
    return (1+np.cos(beta))**2/4
TAB[(2.0, 1)] = _d_4_1

def _d_4_2(beta):
    return np.sin(beta)*(1+np.cos(beta))/2
TAB[(2.0, 2)] = _d_4_2

def _d_4_3(beta):
    return np.sqrt(3/2)/2*np.sin(beta)**2
TAB[(2.0, 3)] = _d_4_3

def _d_4_4(beta):
    return np.sin(beta)*(1-np.cos(beta))/2
TAB[(2.0, 4)] = _d_4_4

def _d_4_5(beta):
    return (1-np.cos(beta))**2/4
TAB[(2.0, 5)] = _d_4_5

def _d_4_6(beta):
    return -np.sin(beta)*(1+np.cos(beta))/2
TAB[(2.0, 6)] = _d_4_6

def _d_4_7(beta):
    cb = np.cos(beta)
    return (2*cb**2+cb-1)/2
TAB[(2.0, 7)] = _d_4_7

def _d_4_8(beta):
    return np.sqrt(3/2)*np.sin(beta)*np.cos(beta)
TAB[(2.0, 8)] = _d_4_8

def _d_4_9(beta):
    cb = np.cos(beta)
    return -(2*cb**2-cb-1)/2
TAB[(2.0, 9)] = _d_4_9

def _d_4_10(beta):
    return np.sin(beta)*(1-np.cos(beta))/2
TAB[(2.0, 10)] = _d_4_10

def _d_4_11(beta):
    return np.sqrt(3/2)/2*np.sin(beta)**2
TAB[(2.0, 11)] = _d_4_11

def _d_4_12(beta):
    return -np.sqrt(3/2)*np.sin(beta)*np.cos(beta)
TAB[(2.0, 12)] = _d_4_12

def _d_4_13(beta):
    return (3*np.cos(beta)**2-1)/2
TAB[(2.0, 13)] = _d_4_13

def _d_4_14(beta):
    return np.sqrt(3/2)*np.sin(beta)*np.cos(beta)
TAB[(2.0, 14)] = _d_4_14

def _d_4_15(beta):
    return np.sqrt(3/2)/2*np.sin(beta)**2
TAB[(2.0, 15)] = _d_4_15

def _d_4_16(beta):
    return -np.sin(beta)*(1-np.cos(beta))/2
TAB[(2.0, 16)] = _d_4_16

def _d_4_17(beta):
    cb = np.cos(beta)
    return -(2*cb**2-cb-1)/2
TAB[(2.0, 17)] = _d_4_17

def _d_4_18(beta):
    return - np.sqrt(3/2)*np.sin(beta)*np.cos(beta)
TAB[(2.0, 18)] = _d_4_18

def _d_4_19(beta):
    cb = np.cos(beta)
    return (2*cb**2+cb-1)/2
TAB[(2.0, 19)] = _d_4_19

def _d_4_20(beta):
    return np.sin(beta)*(1+np.cos(beta))/2
TAB[(2.0, 20)] = _d_4_20

def _d_4_21(beta):
    return (1-np.cos(beta))**2/4
TAB[(2.0, 21)] = _d_4_21

def _d_4_22(beta):
    return -np.sin(beta)*(1-np.cos(beta))/2
TAB[(2.0, 22)] = _d_4_22

def _d_4_23(beta):
    return np.sqrt(3/2)/2*np.sin(beta)**2
TAB[(2.0, 23)] = _d_4_23

def _d_4_24(beta):
    return -np.sin(beta)*(1+np.cos(beta))/2
TAB[(2.0, 24)] = _d_4_24

def _d_4_25(beta):
    return (1+np.cos(beta))**2/4
TAB[(2.0, 25)] = _d_4_25

def _d_5_1(beta):
    return np.cos(beta/2)**5
TAB[(2.5, 1)] = _d_5_1

def _d_5_2(beta):
    return np.sqrt(5)*np.sin(beta/2)*np.cos(beta/2)**4
TAB[(2.5, 2)] = _d_5_2

def _d_5_3(beta):
    return np.sqrt(10)*np.sin(beta/2)**2*np.cos(beta/2)**3
TAB[(2.5, 3)] = _d_5_3

def _d_5_4(beta):
    return np.sqrt(10)*np.sin(beta/2)**3*np.cos(beta/2)**2
TAB[(2.5, 4)] = _d_5_4

def _d_5_5(beta):
    return np.sqrt(5)*np.sin(beta/2)**4*np.cos(beta/2)
TAB[(2.5, 5)] = _d_5_5

def _d_5_6(beta):
    return np.sin(beta/2)**5
TAB[(2.5, 6)] = _d_5_6

def _d_5_7(beta):
    return -np.sqrt(5)*np.sin(beta/2)*np.cos(beta/2)**4
TAB[(2.5, 7)] = _d_5_7

def _d_5_8(beta):
    return np.cos(beta/2)**3*(1-5*np.sin(beta/2)**2)
TAB[(2.5, 8)] = _d_5_8

def _d_5_9(beta):
    return np.sqrt(2)*np.sin(beta/2)*np.cos(beta/2)**2*(2-5*np.sin(beta/2)**2)
TAB[(2.5, 9)] = _d_5_9

def _d_5_10(beta):
    return -np.sqrt(2)*np.sin(beta/2)**2*np.cos(beta/2)*(2-5*np.cos(beta/2)**2)
TAB[(2.5, 10)] = _d_5_10

def _d_5_11(beta):
    return -np.sin(beta/2)**3*(1-5*np.cos(beta/2)**2)
TAB[(2.5, 11)] = _d_5_11

def _d_5_12(beta):
    return np.sqrt(5)*np.sin(beta/2)**4*np.cos(beta/2)
TAB[(2.5, 12)] = _d_5_12

def _d_5_13(beta):
    return np.sqrt(10)*np.sin(beta/2)**2*np.cos(beta/2)**3
TAB[(2.5, 13)] = _d_5_13

def _d_5_14(beta):
    return -np.sqrt(2)*np.sin(beta/2)*np.cos(beta/2)**2*(2-5*np.sin(beta/2)**2)
TAB[(2.5, 14)] = _d_5_14

def _d_5_15(beta):
    return np.cos(beta/2)*(3-12*np.cos(beta/2)**2+10*np.cos(beta/2)**4)
TAB[(2.5, 15)] = _d_5_15

def _d_5_16(beta):
    return np.sin(beta/2)*(3-12*np.sin(beta/2)**2+10*np.sin(beta/2)**4)
TAB[(2.5, 16)] = _d_5_16

def _d_5_17(beta):
    return -np.sqrt(2)*np.sin(beta/2)**2*np.cos(beta/2)*(2-5*np.cos(beta/2)**2)
TAB[(2.5, 17)] = _d_5_17

def _d_5_18(beta):
    return np.sqrt(10)*np.sin(beta/2)**3*np.cos(beta/2)**2
TAB[(2.5, 18)] = _d_5_18

def _d_5_19(beta):
    return -np.sqrt(10)*np.sin(beta/2)**3*np.cos(beta/2)**2
TAB[(2.5, 19)] = _d_5_19

def _d_5_20(beta):
    return -np.sqrt(2)*np.sin(beta/2)**2*np.cos(beta/2)*(2-5*np.cos(beta/2)**2)
TAB[(2.5, 20)] = _d_5_20

def _d_5_21(beta):
    return -np.sin(beta/2)*(3-12*np.sin(beta/2)**2+10*np.sin(beta/2)**4)
TAB[(2.5, 21)] = _d_5_21

def _d_5_22(beta):
    return np.cos(beta/2)*(3-12*np.cos(beta/2)**2+10*np.cos(beta/2)**4)
TAB[(2.5, 22)] = _d_5_22

def _d_5_23(beta):
    return np.sqrt(2)*np.sin(beta/2)*np.cos(beta/2)**2*(2-5*np.sin(beta/2)**2)
TAB[(2.5, 23)] = _d_5_23

def _d_5_24(beta):
    return np.sqrt(10)*np.sin(beta/2)**2*np.cos(beta/2)**3
TAB[(2.5, 24)] = _d_5_24

def _d_5_25(beta):
    return np.sqrt(5)*np.sin(beta/2)**4*np.cos(beta/2)
TAB[(2.5, 25)] = _d_5_25

def _d_5_26(beta):
    return np.sin(beta/2)**3*(1-5*np.cos(beta/2)**2)
TAB[(2.5, 26)] = _d_5_26

def _d_5_27(beta):
    return -np.sqrt(2)*np.sin(beta/2)**2*np.cos(beta/2)*(2-5*np.cos(beta/2)**2)
TAB[(2.5, 27)] = _d_5_27

def _d_5_28(beta):
    return -np.sqrt(2)*np.sin(beta/2)*np.cos(beta/2)**2*(2-5*np.sin(beta/2)**2)
TAB[(2.5, 28)] = _d_5_28

def _d_5_29(beta):
    return np.cos(beta/2)**3*(1-5*np.sin(beta/2)**2)
TAB[(2.5, 29)] = _d_5_29

def _d_5_30(beta):
    return np.sqrt(5)*np.sin(beta/2)*np.cos(beta/2)**4
TAB[(2.5, 30)] = _d_5_30

def _d_5_31(beta):
    return -np.sin(beta/2)**5
TAB[(2.5, 31)] = _d_5_31

def _d_5_32(beta):
    return np.sqrt(5)*np.sin(beta/2)**4*np.cos(beta/2)
TAB[(2.5, 32)] = _d_5_32

def _d_5_33(beta):
    return -np.sqrt(10)*np.sin(beta/2)**3*np.cos(beta/2)**2
TAB[(2.5, 33)] = _d_5_33

def _d_5_34(beta):
    return np.sqrt(10)*np.sin(beta/2)**2*np.cos(beta/2)**3
TAB[(2.5, 34)] = _d_5_34

def _d_5_35(beta):
    return -np.sqrt(5)*np.sin(beta/2)*np.cos(beta/2)**4
TAB[(2.5, 35)] = _d_5_35

def _d_5_36(beta):
    return np.cos(beta/2)**5
TAB[(2.5, 36)] = _d_5_36

def _d_6_1(beta):
    return (1+np.cos(beta))**3/8
TAB[(3.0, 1)] = _d_6_1

def _d_6_2(beta):
    return np.sqrt(6)/8*np.sin(beta)*(1+np.cos(beta))**2
TAB[(3.0, 2)] = _d_6_2

def _d_6_3(beta):
    return np.sqrt(15)/8*np.sin(beta)**2*(1+np.cos(beta))
TAB[(3.0, 3)] = _d_6_3

def _d_6_4(beta):
    return np.sqrt(5)/4*np.sin(beta)**3
TAB[(3.0, 4)] = _d_6_4

def _d_6_5(beta):
    return np.sqrt(15)/8*np.sin(beta)**2*(1-np.cos(beta))
TAB[(3.0, 5)] = _d_6_5

def _d_6_6(beta):
    return np.sqrt(6)/8*np.sin(beta)*(1-np.cos(beta))**2
TAB[(3.0, 6)] = _d_6_6

def _d_6_7(beta):
    return (1-np.cos(beta))**3/8
TAB[(3.0, 7)] = _d_6_7

def _d_6_8(beta):
    return -np.sqrt(6)/8*np.sin(beta)*(1+np.cos(beta))**2
TAB[(3.0, 8)] = _d_6_8

def _d_6_9(beta):
    return -1/4*(1+np.cos(beta))**2*(2-3*np.cos(beta))
TAB[(3.0, 9)] = _d_6_9

def _d_6_10(beta):
    return -np.sqrt(10)/8*np.sin(beta)*(1-2*np.cos(beta)-3*np.cos(beta)**2)
TAB[(3.0, 10)] = _d_6_10

def _d_6_11(beta):
    return np.sqrt(30)/4*np.sin(beta)**2*np.cos(beta)
TAB[(3.0, 11)] = _d_6_11

def _d_6_12(beta):
    return np.sqrt(10)/8*np.sin(beta)*(1+2*np.cos(beta)-3*np.cos(beta)**2)
TAB[(3.0, 12)] = _d_6_12

def _d_6_13(beta):
    return 1/4*(1-np.cos(beta))**2*(2+3*np.cos(beta))
TAB[(3.0, 13)] = _d_6_13

def _d_6_14(beta):
    return np.sqrt(6)/8*np.sin(beta)*(1-np.cos(beta))**2
TAB[(3.0, 14)] = _d_6_14

def _d_6_15(beta):
    return np.sqrt(15)/8*np.sin(beta)**2*(1+np.cos(beta))
TAB[(3.0, 15)] = _d_6_15

def _d_6_16(beta):
    return np.sqrt(10)/8*np.sin(beta)*(1-2*np.cos(beta)-3*np.cos(beta)**2)
TAB[(3.0, 16)] = _d_6_16

def _d_6_17(beta):
    return -1/8*(1+np.cos(beta))*(1+10*np.cos(beta)-15*np.cos(beta)**2)
TAB[(3.0, 17)] = _d_6_17

def _d_6_18(beta):
    return -np.sqrt(3)/4*np.sin(beta)*(1-5*np.cos(beta)**2)
TAB[(3.0, 18)] = _d_6_18

def _d_6_19(beta):
    return -1/8*(1-np.cos(beta))*(1-10*np.cos(beta)-15*np.cos(beta)**2)
TAB[(3.0, 19)] = _d_6_19

def _d_6_20(beta):
    return np.sqrt(10)/8*np.sin(beta)*(1+2*np.cos(beta)-3*np.cos(beta)**2)
TAB[(3.0, 20)] = _d_6_20

def _d_6_21(beta):
    return np.sqrt(15)/8*np.sin(beta)**2*(1-np.cos(beta))
TAB[(3.0, 21)] = _d_6_21

def _d_6_22(beta):
    return -np.sqrt(5)/4*np.sin(beta)**3
TAB[(3.0, 22)] = _d_6_22

def _d_6_23(beta):
    return np.sqrt(30)/4*np.sin(beta)**2*np.cos(beta)
TAB[(3.0, 23)] = _d_6_23

def _d_6_24(beta):
    return np.sqrt(3)/4*np.sin(beta)*(1-5*np.cos(beta)**2)
TAB[(3.0, 24)] = _d_6_24

def _d_6_25(beta):
    return -1/2*np.cos(beta)*(3-5*np.cos(beta)**2)
TAB[(3.0, 25)] = _d_6_25

def _d_6_26(beta):
    return -np.sqrt(3)/4*np.sin(beta)*(1-5*np.cos(beta)**2)
TAB[(3.0, 26)] = _d_6_26

def _d_6_27(beta):
    return np.sqrt(30)/4*np.sin(beta)**2*np.cos(beta)
TAB[(3.0, 27)] = _d_6_27

def _d_6_28(beta):
    return np.sqrt(5)/4*np.sin(beta)**3
TAB[(3.0, 28)] = _d_6_28

def _d_6_29(beta):
    return np.sqrt(15)/8*np.sin(beta)**2*(1-np.cos(beta))
TAB[(3.0, 29)] = _d_6_29

def _d_6_30(beta):
    return -np.sqrt(10)/8*np.sin(beta)*(1+2*np.cos(beta)-3*np.cos(beta)**2)
TAB[(3.0, 30)] = _d_6_30

def _d_6_31(beta):
    return -1/8*(1-np.cos(beta))*(1-10*np.cos(beta)-15*np.cos(beta)**2)
TAB[(3.0, 31)] = _d_6_31

def _d_6_32(beta):
    return np.sqrt(3)/4*np.sin(beta)*(1-5*np.cos(beta)**2)
TAB[(3.0, 32)] = _d_6_32

def _d_6_33(beta):
    return -1/8*(1+np.cos(beta))*(1+10*np.cos(beta)-15*np.cos(beta)**2)
TAB[(3.0, 33)] = _d_6_33

def _d_6_34(beta):
    return -np.sqrt(10)/8*np.sin(beta)*(1-2*np.cos(beta)-3*np.cos(beta)**2)
TAB[(3.0, 34)] = _d_6_34

def _d_6_35(beta):
    return np.sqrt(15)/8*np.sin(beta)**2*(1+np.cos(beta))
TAB[(3.0, 35)] = _d_6_35

def _d_6_36(beta):
    return -np.sqrt(6)/8*np.sin(beta)*(1-np.cos(beta))**2
TAB[(3.0, 36)] = _d_6_36

def _d_6_37(beta):
    return 1/4*(1-np.cos(beta))**2*(2+3*np.cos(beta))
TAB[(3.0, 37)] = _d_6_37

def _d_6_38(beta):
    return -np.sqrt(10)/8*np.sin(beta)*(1+2*np.cos(beta)-3*np.cos(beta)**2)
TAB[(3.0, 38)] = _d_6_38

def _d_6_39(beta):
    return np.sqrt(30)/4*np.sin(beta)**2*np.cos(beta)
TAB[(3.0, 39)] = _d_6_39

def _d_6_40(beta):
    return np.sqrt(10)/8*np.sin(beta)*(1-2*np.cos(beta)-3*np.cos(beta)**2)
TAB[(3.0, 40)] = _d_6_40

def _d_6_41(beta):
    return -1/4*(1+np.cos(beta))**2*(2-3*np.cos(beta))
TAB[(3.0, 41)] = _d_6_41

def _d_6_42(beta):
    return np.sqrt(6)/8*np.sin(beta)*(1+np.cos(beta))**2
TAB[(3.0, 42)] = _d_6_42

def _d_6_43(beta):
    return (1-np.cos(beta))**3/8
TAB[(3.0, 43)] = _d_6_43

def _d_6_44(beta):
    return -np.sqrt(6)/8*np.sin(beta)*(1-np.cos(beta))**2
TAB[(3.0, 44)] = _d_6_44

def _d_6_45(beta):
    return np.sqrt(15)/8*np.sin(beta)**2*(1-np.cos(beta))
TAB[(3.0, 45)] = _d_6_45

def _d_6_46(beta):
    return -np.sqrt(5)/4*np.sin(beta)**3
TAB[(3.0, 46)] = _d_6_46

def _d_6_47(beta):
    return np.sqrt(15)/8*np.sin(beta)**2*(1+np.cos(beta))
TAB[(3.0, 47)] = _d_6_47

def _d_6_48(beta):
    return -np.sqrt(6)/8*np.sin(beta)*(1+np.cos(beta))**2
TAB[(3.0, 48)] = _d_6_48

def _d_6_49(beta):
    return (1+np.cos(beta))**3/8
TAB[(3.0, 49)] = _d_6_49

def _d_7_1(beta):
    return np.cos(beta/2)**7
TAB[(3.5, 1)] = _d_7_1

def _d_7_2(beta):
    return np.sqrt(7)*np.cos(beta/2)**6*np.sin(beta/2)
TAB[(3.5, 2)] = _d_7_2

def _d_7_3(beta):
    return np.sqrt(21)*np.cos(beta/2)**5*np.sin(beta/2)**2
TAB[(3.5, 3)] = _d_7_3

def _d_7_4(beta):
    return np.sqrt(35)*np.cos(beta/2)**4*np.sin(beta/2)**3
TAB[(3.5, 4)] = _d_7_4

def _d_7_5(beta):
    return np.sqrt(35)*np.cos(beta/2)**3*np.sin(beta/2)**4
TAB[(3.5, 5)] = _d_7_5

def _d_7_6(beta):
    return np.sqrt(21)*np.cos(beta/2)**2*np.sin(beta/2)**5
TAB[(3.5, 6)] = _d_7_6

def _d_7_7(beta):
    return np.sqrt(7)*np.cos(beta/2)*np.sin(beta/2)**6
TAB[(3.5, 7)] = _d_7_7

def _d_7_8(beta):
    return np.sin(beta/2)**7
TAB[(3.5, 8)] = _d_7_8

def _d_7_9(beta):
    return -np.sqrt(7)*np.cos(beta/2)**6*np.sin(beta/2)
TAB[(3.5, 9)] = _d_7_9

def _d_7_10(beta):
    return np.cos(beta/2)**5*(1-7*np.sin(beta/2)**2)
TAB[(3.5, 10)] = _d_7_10

def _d_7_11(beta):
    return np.sqrt(3)*np.cos(beta/2)**4*np.sin(beta/2)*(2-7*np.sin(beta/2)**2)
TAB[(3.5, 11)] = _d_7_11

def _d_7_12(beta):
    return np.sqrt(5)*np.cos(beta/2)**3*np.sin(beta/2)**2*(3-7*np.sin(beta/2)**2)
TAB[(3.5, 12)] = _d_7_12

def _d_7_13(beta):
    return -np.sqrt(5)*np.cos(beta/2)**2*np.sin(beta/2)**3*(3-7*np.cos(beta/2)**2)
TAB[(3.5, 13)] = _d_7_13

def _d_7_14(beta):
    return -np.sqrt(3)*np.cos(beta/2)*np.sin(beta/2)**4*(2-7*np.cos(beta/2)**2)
TAB[(3.5, 14)] = _d_7_14

def _d_7_15(beta):
    return -np.sin(beta/2)**5*(1-7*np.cos(beta/2)**2)
TAB[(3.5, 15)] = _d_7_15

def _d_7_16(beta):
    return np.sqrt(7)*np.cos(beta/2)*np.sin(beta/2)**6
TAB[(3.5, 16)] = _d_7_16

def _d_7_17(beta):
    return np.sqrt(21)*np.cos(beta/2)**5*np.sin(beta/2)**2
TAB[(3.5, 17)] = _d_7_17

def _d_7_18(beta):
    return -np.sqrt(3)*np.cos(beta/2)**4*np.sin(beta/2)*(2-7*np.sin(beta/2)**2)
TAB[(3.5, 18)] = _d_7_18

def _d_7_19(beta):
    return np.cos(beta/2)**3*(10-30*np.cos(beta/2)**2+21*np.cos(beta/2)**4)
TAB[(3.5, 19)] = _d_7_19

def _d_7_20(beta):
    return np.sqrt(15)*np.cos(beta/2)**2*np.sin(beta/2)*(2-8*np.cos(beta/2)**2+7*np.cos(beta/2)**4)
TAB[(3.5, 20)] = _d_7_20

def _d_7_21(beta):
    return np.sqrt(15)*np.cos(beta/2)*np.sin(beta/2)**2*(2-8*np.sin(beta/2)**2+7*np.sin(beta/2)**4)
TAB[(3.5, 21)] = _d_7_21

def _d_7_22(beta):
    return np.sin(beta/2)**3*(10-30*np.sin(beta/2)**2+21*np.sin(beta/2)**4)
TAB[(3.5, 22)] = _d_7_22

def _d_7_23(beta):
    return -np.sqrt(3)*np.cos(beta/2)*np.sin(beta/2)**4*(2-7*np.cos(beta/2)**2)
TAB[(3.5, 23)] = _d_7_23

def _d_7_24(beta):
    return np.sqrt(21)*np.cos(beta/2)**2*np.sin(beta/2)**5
TAB[(3.5, 24)] = _d_7_24

def _d_7_25(beta):
    return -np.sqrt(35)*np.cos(beta/2)**4*np.sin(beta/2)**3
TAB[(3.5, 25)] = _d_7_25

def _d_7_26(beta):
    return np.sqrt(5)*np.cos(beta/2)**3*np.sin(beta/2)**2*(3-7*np.sin(beta/2)**2)
TAB[(3.5, 26)] = _d_7_26

def _d_7_27(beta):
    return -np.sqrt(15)*np.cos(beta/2)**2*np.sin(beta/2)*(2-8*np.cos(beta/2)**2+7*np.cos(beta/2)**4)
TAB[(3.5, 27)] = _d_7_27

def _d_7_28(beta):
    return -np.cos(beta/2)*(4-30*np.cos(beta/2)**2+60*np.cos(beta/2)**4-35*np.cos(beta/2)**6)
TAB[(3.5, 28)] = _d_7_28

def _d_7_29(beta):
    return np.sin(beta/2)*(4-30*np.sin(beta/2)**2+60*np.sin(beta/2)**4-35*np.sin(beta/2)**6)
TAB[(3.5, 29)] = _d_7_29

def _d_7_30(beta):
    return np.sqrt(15)*np.cos(beta/2)*np.sin(beta/2)**2*(2-8*np.sin(beta/2)**2+7*np.sin(beta/2)**4)
TAB[(3.5, 30)] = _d_7_30

def _d_7_31(beta):
    return -np.sqrt(5)*np.cos(beta/2)**2*np.sin(beta/2)**3*(3-7*np.cos(beta/2)**2)
TAB[(3.5, 31)] = _d_7_31

def _d_7_32(beta):
    return np.sqrt(35)*np.cos(beta/2)**3*np.sin(beta/2)**4
TAB[(3.5, 32)] = _d_7_32

def _d_7_33(beta):
    return np.sqrt(35)*np.cos(beta/2)**3*np.sin(beta/2)**4
TAB[(3.5, 33)] = _d_7_33

def _d_7_34(beta):
    return np.sqrt(5)*np.cos(beta/2)**2*np.sin(beta/2)**3*(3-7*np.cos(beta/2)**2)
TAB[(3.5, 34)] = _d_7_34

def _d_7_35(beta):
    return np.sqrt(15)*np.cos(beta/2)*np.sin(beta/2)**2*(2-8*np.sin(beta/2)**2+7*np.sin(beta/2)**4)
TAB[(3.5, 35)] = _d_7_35

def _d_7_36(beta):
    return -np.sin(beta/2)*(4-30*np.sin(beta/2)**2+60*np.sin(beta/2)**4-35*np.sin(beta/2)**6)
TAB[(3.5, 36)] = _d_7_36

def _d_7_37(beta):
    return -np.cos(beta/2)*(4-30*np.cos(beta/2)**2+60*np.cos(beta/2)**4-35*np.cos(beta/2)**6)
TAB[(3.5, 37)] = _d_7_37

def _d_7_38(beta):
    return np.sqrt(15)*np.cos(beta/2)**2*np.sin(beta/2)*(2-8*np.cos(beta/2)**2+7*np.cos(beta/2)**4)
TAB[(3.5, 38)] = _d_7_38

def _d_7_39(beta):
    return np.sqrt(5)*np.cos(beta/2)**3*np.sin(beta/2)**2*(3-7*np.sin(beta/2)**2)
TAB[(3.5, 39)] = _d_7_39

def _d_7_40(beta):
    return np.sqrt(35)*np.cos(beta/2)**4*np.sin(beta/2)**3
TAB[(3.5, 40)] = _d_7_40

def _d_7_41(beta):
    return -np.sqrt(21)*np.cos(beta/2)**2*np.sin(beta/2)**5
TAB[(3.5, 41)] = _d_7_41

def _d_7_42(beta):
    return -np.sqrt(3)*np.cos(beta/2)*np.sin(beta/2)**4*(2-7*np.cos(beta/2)**2)
TAB[(3.5, 42)] = _d_7_42

def _d_7_43(beta):
    return -np.sin(beta/2)**3*(10-30*np.sin(beta/2)**2+21*np.sin(beta/2)**4)
TAB[(3.5, 43)] = _d_7_43

def _d_7_44(beta):
    return np.sqrt(15)*np.cos(beta/2)*np.sin(beta/2)**2*(2-8*np.sin(beta/2)**2+7*np.sin(beta/2)**4)
TAB[(3.5, 44)] = _d_7_44

def _d_7_45(beta):
    return -np.sqrt(15)*np.cos(beta/2)**2*np.sin(beta/2)*(2-8*np.cos(beta/2)**2+7*np.cos(beta/2)**4)
TAB[(3.5, 45)] = _d_7_45

def _d_7_46(beta):
    return np.cos(beta/2)**3*(10-30*np.cos(beta/2)**2+21*np.cos(beta/2)**4)
TAB[(3.5, 46)] = _d_7_46

def _d_7_47(beta):
    return np.sqrt(3)*np.cos(beta/2)**4*np.sin(beta/2)*(2-7*np.sin(beta/2)**2)
TAB[(3.5, 47)] = _d_7_47

def _d_7_48(beta):
    return np.sqrt(21)*np.cos(beta/2)**5*np.sin(beta/2)**2
TAB[(3.5, 48)] = _d_7_48

def _d_7_49(beta):
    return np.sqrt(7)*np.cos(beta/2)*np.sin(beta/2)**6
TAB[(3.5, 49)] = _d_7_49

def _d_7_50(beta):
    return np.sin(beta/2)**5*(1-7*np.cos(beta/2)**2)
TAB[(3.5, 50)] = _d_7_50

def _d_7_51(beta):
    return -np.sqrt(3)*np.cos(beta/2)*np.sin(beta/2)**4*(2-7*np.cos(beta/2)**2)
TAB[(3.5, 51)] = _d_7_51

def _d_7_52(beta):
    return np.sqrt(5)*np.cos(beta/2)**2*np.sin(beta/2)**3*(3-7*np.cos(beta/2)**2)
TAB[(3.5, 52)] = _d_7_52

def _d_7_53(beta):
    return np.sqrt(5)*np.cos(beta/2)**3*np.sin(beta/2)**2*(3-7*np.sin(beta/2)**2)
TAB[(3.5, 53)] = _d_7_53

def _d_7_54(beta):
    return -np.sqrt(3)*np.cos(beta/2)**4*np.sin(beta/2)*(2-7*np.sin(beta/2)**2)
TAB[(3.5, 54)] = _d_7_54

def _d_7_55(beta):
    return np.cos(beta/2)**5*(1-7*np.sin(beta/2)**2)
TAB[(3.5, 55)] = _d_7_55

def _d_7_56(beta):
    return np.sqrt(7)*np.cos(beta/2)**6*np.sin(beta/2)
TAB[(3.5, 56)] = _d_7_56

def _d_7_57(beta):
    return -np.sin(beta/2)**7
TAB[(3.5, 57)] = _d_7_57

def _d_7_58(beta):
    return np.sqrt(7)*np.cos(beta/2)*np.sin(beta/2)**6
TAB[(3.5, 58)] = _d_7_58

def _d_7_59(beta):
    return -np.sqrt(21)*np.cos(beta/2)**2*np.sin(beta/2)**5
TAB[(3.5, 59)] = _d_7_59

def _d_7_60(beta):
    return np.sqrt(35)*np.cos(beta/2)**3*np.sin(beta/2)**4
TAB[(3.5, 60)] = _d_7_60

def _d_7_61(beta):
    return -np.sqrt(35)*np.cos(beta/2)**4*np.sin(beta/2)**3
TAB[(3.5, 61)] = _d_7_61

def _d_7_62(beta):
    return np.sqrt(21)*np.cos(beta/2)**5*np.sin(beta/2)**2
TAB[(3.5, 62)] = _d_7_62

def _d_7_63(beta):
    return -np.sqrt(7)*np.cos(beta/2)**6*np.sin(beta/2)
TAB[(3.5, 63)] = _d_7_63

def _d_7_64(beta):
    return np.cos(beta/2)**7
TAB[(3.5, 64)] = _d_7_64

def _d_8_1(beta):
    return (1+np.cos(beta))**4/16
TAB[(4.0, 1)] = _d_8_1

def _d_8_2(beta):
    return np.sqrt(2)/8*np.sin(beta)*(1+np.cos(beta))**3
TAB[(4.0, 2)] = _d_8_2

def _d_8_3(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1+np.cos(beta))**2
TAB[(4.0, 3)] = _d_8_3

def _d_8_4(beta):
    return np.sqrt(14)/8*np.sin(beta)**3*(1+np.cos(beta))
TAB[(4.0, 4)] = _d_8_4

def _d_8_5(beta):
    return np.sqrt(70)/16*np.sin(beta)**4
TAB[(4.0, 5)] = _d_8_5

def _d_8_6(beta):
    return np.sqrt(14)/8*np.sin(beta)**3*(1-np.cos(beta))
TAB[(4.0, 6)] = _d_8_6

def _d_8_7(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1-np.cos(beta))**2
TAB[(4.0, 7)] = _d_8_7

def _d_8_8(beta):
    return np.sqrt(2)/8*np.sin(beta)*(1-np.cos(beta))**3
TAB[(4.0, 8)] = _d_8_8

def _d_8_9(beta):
    return (1-np.cos(beta))**4/16
TAB[(4.0, 9)] = _d_8_9

def _d_8_10(beta):
    return -np.sqrt(2)/8*np.sin(beta)*(1+np.cos(beta))**3
TAB[(4.0, 10)] = _d_8_10

def _d_8_11(beta):
    return -1/8*(1+np.cos(beta))**3*(3-4*np.cos(beta))
TAB[(4.0, 11)] = _d_8_11

def _d_8_12(beta):
    return -np.sqrt(14)/8*np.sin(beta)*(1+np.cos(beta))**2*(1-2*np.cos(beta))
TAB[(4.0, 12)] = _d_8_12

def _d_8_13(beta):
    return -np.sqrt(7)/8*np.sin(beta)**2*(1+np.cos(beta))*(1-4*np.cos(beta))
TAB[(4.0, 13)] = _d_8_13

def _d_8_14(beta):
    return np.sqrt(35)/4*np.sin(beta)**3*np.cos(beta)
TAB[(4.0, 14)] = _d_8_14

def _d_8_15(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1-np.cos(beta))*(1+4*np.cos(beta))
TAB[(4.0, 15)] = _d_8_15

def _d_8_16(beta):
    return np.sqrt(14)/8*np.sin(beta)*(1-np.cos(beta))**2*(1+2*np.cos(beta))
TAB[(4.0, 16)] = _d_8_16

def _d_8_17(beta):
    return 1/8*(1-np.cos(beta))**3*(3+4*np.cos(beta))
TAB[(4.0, 17)] = _d_8_17

def _d_8_18(beta):
    return np.sqrt(2)/8*np.sin(beta)*(1-np.cos(beta))**3
TAB[(4.0, 18)] = _d_8_18

def _d_8_19(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1+np.cos(beta))**2
TAB[(4.0, 19)] = _d_8_19

def _d_8_20(beta):
    return np.sqrt(14)/8*np.sin(beta)*(1+np.cos(beta))**2*(1-2*np.cos(beta))
TAB[(4.0, 20)] = _d_8_20

def _d_8_21(beta):
    return 1/4*(1+np.cos(beta))**2*(1-7*np.cos(beta)+7*np.cos(beta)**2)
TAB[(4.0, 21)] = _d_8_21

def _d_8_22(beta):
    return -np.sqrt(2)/8*np.sin(beta)*(1+np.cos(beta))*(1+7*np.cos(beta)-14*np.cos(beta)**2)
TAB[(4.0, 22)] = _d_8_22

def _d_8_23(beta):
    return -np.sqrt(10)/8*np.sin(beta)**2*(1-7*np.cos(beta)**2)
TAB[(4.0, 23)] = _d_8_23

def _d_8_24(beta):
    return -np.sqrt(2)/8*np.sin(beta)*(1-np.cos(beta))*(1-7*np.cos(beta)-14*np.cos(beta)**2)
TAB[(4.0, 24)] = _d_8_24

def _d_8_25(beta):
    return 1/4*(1-np.cos(beta))**2*(1+7*np.cos(beta)+7*np.cos(beta)**2)
TAB[(4.0, 25)] = _d_8_25

def _d_8_26(beta):
    return np.sqrt(14)/8*np.sin(beta)*(1-np.cos(beta))**2*(1+2*np.cos(beta))
TAB[(4.0, 26)] = _d_8_26

def _d_8_27(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1-np.cos(beta))**2
TAB[(4.0, 27)] = _d_8_27

def _d_8_28(beta):
    return -np.sqrt(14)/8*np.sin(beta)**3*(1+np.cos(beta))
TAB[(4.0, 28)] = _d_8_28

def _d_8_29(beta):
    return -np.sqrt(7)/8*np.sin(beta)**2*(1+np.cos(beta))*(1-4*np.cos(beta))
TAB[(4.0, 29)] = _d_8_29

def _d_8_30(beta):
    return np.sqrt(2)/8*np.sin(beta)*(1+np.cos(beta))*(1+7*np.cos(beta)-14*np.cos(beta)**2)
TAB[(4.0, 30)] = _d_8_30

def _d_8_31(beta):
    return 1/8*(1+np.cos(beta))*(3-6*np.cos(beta)-21*np.cos(beta)**2+28*np.cos(beta)**3)
TAB[(4.0, 31)] = _d_8_31

def _d_8_32(beta):
    return -np.sqrt(5)/4*np.sin(beta)*np.cos(beta)*(3-7*np.cos(beta)**2)
TAB[(4.0, 32)] = _d_8_32

def _d_8_33(beta):
    return -1/8*(1-np.cos(beta))*(3+6*np.cos(beta)-21*np.cos(beta)**2-28*np.cos(beta)**3)
TAB[(4.0, 33)] = _d_8_33

def _d_8_34(beta):
    return -np.sqrt(2)/8*np.sin(beta)*(1-np.cos(beta))*(1-7*np.cos(beta)-14*np.cos(beta)**2)
TAB[(4.0, 34)] = _d_8_34

def _d_8_35(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1-np.cos(beta))*(1+4*np.cos(beta))
TAB[(4.0, 35)] = _d_8_35

def _d_8_36(beta):
    return np.sqrt(14)/8*np.sin(beta)**3*(1-np.cos(beta))
TAB[(4.0, 36)] = _d_8_36

def _d_8_37(beta):
    return np.sqrt(70)/16*np.sin(beta)**4
TAB[(4.0, 37)] = _d_8_37

def _d_8_38(beta):
    return -np.sqrt(35)/4*np.sin(beta)**3*np.cos(beta)
TAB[(4.0, 38)] = _d_8_38

def _d_8_39(beta):
    return -np.sqrt(10)/8*np.sin(beta)**2*(1-7*np.cos(beta)**2)
TAB[(4.0, 39)] = _d_8_39

def _d_8_40(beta):
    return np.sqrt(5)/4*np.sin(beta)*np.cos(beta)*(3-7*np.cos(beta)**2)
TAB[(4.0, 40)] = _d_8_40

def _d_8_41(beta):
    return 1/8*(3-30*np.cos(beta)**2+35*np.cos(beta)**4)
TAB[(4.0, 41)] = _d_8_41

def _d_8_42(beta):
    return -np.sqrt(5)/4*np.sin(beta)*np.cos(beta)*(3-7*np.cos(beta)**2)
TAB[(4.0, 42)] = _d_8_42

def _d_8_43(beta):
    return -np.sqrt(10)/8*np.sin(beta)**2*(1-7*np.cos(beta)**2)
TAB[(4.0, 43)] = _d_8_43

def _d_8_44(beta):
    return np.sqrt(35)/4*np.sin(beta)**3*np.cos(beta)
TAB[(4.0, 44)] = _d_8_44

def _d_8_45(beta):
    return np.sqrt(70)/16*np.sin(beta)**4
TAB[(4.0, 45)] = _d_8_45

def _d_8_46(beta):
    return -np.sqrt(14)/8*np.sin(beta)**3*(1-np.cos(beta))
TAB[(4.0, 46)] = _d_8_46

def _d_8_47(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1-np.cos(beta))*(1+4*np.cos(beta))
TAB[(4.0, 47)] = _d_8_47

def _d_8_48(beta):
    return np.sqrt(2)/8*np.sin(beta)*(1-np.cos(beta))*(1-7*np.cos(beta)-14*np.cos(beta)**2)
TAB[(4.0, 48)] = _d_8_48

def _d_8_49(beta):
    return -1/8*(1-np.cos(beta))*(3+6*np.cos(beta)-21*np.cos(beta)**2-28*np.cos(beta)**3)
TAB[(4.0, 49)] = _d_8_49

def _d_8_50(beta):
    return np.sqrt(5)/4*np.sin(beta)*np.cos(beta)*(3-7*np.cos(beta)**2)
TAB[(4.0, 50)] = _d_8_50

def _d_8_51(beta):
    return 1/8*(1+np.cos(beta))*(3-6*np.cos(beta)-21*np.cos(beta)**2+28*np.cos(beta)**3)
TAB[(4.0, 51)] = _d_8_51

def _d_8_52(beta):
    return -np.sqrt(2)/8*np.sin(beta)*(1+np.cos(beta))*(1+7*np.cos(beta)-14*np.cos(beta)**2)
TAB[(4.0, 52)] = _d_8_52

def _d_8_53(beta):
    return -np.sqrt(7)/8*np.sin(beta)**2*(1+np.cos(beta))*(1-4*np.cos(beta))
TAB[(4.0, 53)] = _d_8_53

def _d_8_54(beta):
    return np.sqrt(14)/8*np.sin(beta)**3*(1+np.cos(beta))
TAB[(4.0, 54)] = _d_8_54

def _d_8_55(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1-np.cos(beta))**2
TAB[(4.0, 55)] = _d_8_55

def _d_8_56(beta):
    return -np.sqrt(14)/8*np.sin(beta)*(1-np.cos(beta))**2*(1+2*np.cos(beta))
TAB[(4.0, 56)] = _d_8_56

def _d_8_57(beta):
    return 1/4*(1-np.cos(beta))**2*(1+7*np.cos(beta)+7*np.cos(beta)**2)
TAB[(4.0, 57)] = _d_8_57

def _d_8_58(beta):
    return np.sqrt(2)/8*np.sin(beta)*(1-np.cos(beta))*(1-7*np.cos(beta)-14*np.cos(beta)**2)
TAB[(4.0, 58)] = _d_8_58

def _d_8_59(beta):
    return -np.sqrt(10)/8*np.sin(beta)**2*(1-7*np.cos(beta)**2)
TAB[(4.0, 59)] = _d_8_59

def _d_8_60(beta):
    return np.sqrt(2)/8*np.sin(beta)*(1+np.cos(beta))*(1+7*np.cos(beta)-14*np.cos(beta)**2)
TAB[(4.0, 60)] = _d_8_60

def _d_8_61(beta):
    return 1/4*(1+np.cos(beta))**2*(1-7*np.cos(beta)+7*np.cos(beta)**2)
TAB[(4.0, 61)] = _d_8_61

def _d_8_62(beta):
    return -np.sqrt(14)/8*np.sin(beta)*(1+np.cos(beta))**2*(1-2*np.cos(beta))
TAB[(4.0, 62)] = _d_8_62

def _d_8_63(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1+np.cos(beta))**2
TAB[(4.0, 63)] = _d_8_63

def _d_8_64(beta):
    return -np.sqrt(2)/8*np.sin(beta)*(1-np.cos(beta))**3
TAB[(4.0, 64)] = _d_8_64

def _d_8_65(beta):
    return 1/8*(1-np.cos(beta))**3*(3+4*np.cos(beta))
TAB[(4.0, 65)] = _d_8_65

def _d_8_66(beta):
    return -np.sqrt(14)/8*np.sin(beta)*(1-np.cos(beta))**2*(1+2*np.cos(beta))
TAB[(4.0, 66)] = _d_8_66

def _d_8_67(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1-np.cos(beta))*(1+4*np.cos(beta))
TAB[(4.0, 67)] = _d_8_67

def _d_8_68(beta):
    return -np.sqrt(35)/4*np.sin(beta)**3*np.cos(beta)
TAB[(4.0, 68)] = _d_8_68

def _d_8_69(beta):
    return -np.sqrt(7)/8*np.sin(beta)**2*(1+np.cos(beta))*(1-4*np.cos(beta))
TAB[(4.0, 69)] = _d_8_69

def _d_8_70(beta):
    return np.sqrt(14)/8*np.sin(beta)*(1+np.cos(beta))**2*(1-2*np.cos(beta))
TAB[(4.0, 70)] = _d_8_70

def _d_8_71(beta):
    return -1/8*(1+np.cos(beta))**3*(3-4*np.cos(beta))
TAB[(4.0, 71)] = _d_8_71

def _d_8_72(beta):
    return np.sqrt(2)/8*np.sin(beta)*(1+np.cos(beta))**3
TAB[(4.0, 72)] = _d_8_72

def _d_8_73(beta):
    return (1-np.cos(beta))**4/16
TAB[(4.0, 73)] = _d_8_73

def _d_8_74(beta):
    return -np.sqrt(2)/8*np.sin(beta)*(1-np.cos(beta))**3
TAB[(4.0, 74)] = _d_8_74

def _d_8_75(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1-np.cos(beta))**2
TAB[(4.0, 75)] = _d_8_75

def _d_8_76(beta):
    return -np.sqrt(14)/8*np.sin(beta)**3*(1-np.cos(beta))
TAB[(4.0, 76)] = _d_8_76

def _d_8_77(beta):
    return np.sqrt(70)/16*np.sin(beta)**4
TAB[(4.0, 77)] = _d_8_77

def _d_8_78(beta):
    return -np.sqrt(14)/8*np.sin(beta)**3*(1+np.cos(beta))
TAB[(4.0, 78)] = _d_8_78

def _d_8_79(beta):
    return np.sqrt(7)/8*np.sin(beta)**2*(1+np.cos(beta))**2
TAB[(4.0, 79)] = _d_8_79

def _d_8_80(beta):
    return -np.sqrt(2)/8*np.sin(beta)*(1+np.cos(beta))**3
TAB[(4.0, 80)] = _d_8_80

def _d_8_81(beta):
    return (1+np.cos(beta))**4/16
TAB[(4.0, 81)] = _d_8_81

