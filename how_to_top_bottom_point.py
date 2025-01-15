
import numpy as np

"""

Assume we have 1D data for slicing from the leftZero: maxFrames and maxFrames:RightZero. We label these data
as data_range_l and data_range_r, respectively.

Given also the blinkTop and blinkBottom that generated from

        blinkHeight = data(maxFrames) - data(leftZero);
        blinkTop = data(maxFrames) - baseFraction*blinkHeight;
        blinkBottom = data(leftZero) + baseFraction*blinkHeight;
        
        
"""
blinkTop=95.212867041938990
blinkBottom=7.784262036022467


data_range_l=[-3.14431358971710,5.34609700073048,15.3874571164097,25.9301434602452,38.0729234823422,53.7573730625053,
              72.6114031291783,90.6534527072936,102.678666809897,106.141442667679]

data_range_r=[106.141442667679,102.789093239711,96.7204498309708,91.0541025906445,
              86.2916031537046,81.2692020508226,75.0675033319083,67.8974528863945,
              60.5528663188495,53.4850961494706,46.5817615304153,39.6964933458947,
              33.1356096728375,27.4341210244025,22.5990907918291,17.7371629495918,
              11.7813761332345,4.83507448180312,-1.35029656094700]

data_range_l=np.array(data_range_l)
data_range_r=np.array(data_range_r)

# Retrun indices corresponding to the nonzero entries of that array X
blinkTopPoint_l = np.argmin(data_range_l < blinkTop)
assert blinkTopPoint_l==8

blinkBottomPoint_l = np.argmax(data_range_l > blinkBottom) # Expected 3
assert blinkBottomPoint_l==2



blinkTopPoint_r = np.argmax(data_range_r < blinkTop) # Expected 3
assert blinkTopPoint_r==3
blinkBottomPoint_r = np.argmin(data_range_r > blinkBottom) # Expected 17


assert blinkBottomPoint_r==17


