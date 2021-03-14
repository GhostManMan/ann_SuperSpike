from ANNarchy import *
import numpy as np

# function to define the connection pattern for the 'patch pattern' by Michael Teichmann for ANNarchy 4.x
# iterate along the post-synaptic neurons and build a connection to all pre-synaptic neurons wich are in the 'pre-synaptic segmentaion window'
# prePop, postPop -> the pre- and post- synaptic Population-Objects -> geometry must be (w,h,d)
# min_value, max_value -> minimum and maximum values of the uniform distribution to initialize the weights  
# shiftRF -> shift of the pre-synaptic segmentation, in other words, shift of the window what the post-synaptic 'see'
# rfx, rfy -> size or radius of the pre-synaptic segmentation-window
# offsetX, offsetY -> offset for X and Y axis
# delay -> delay in the transmission of the synaptic potential, default = 0
def patch_pattern(prePop, postPop, min_value, max_value, shiftRF, rfX, rfY, offsetX, offsetY, delay=0):

    cc = 0
    synapses = CSR()
    for w_post in range(postPop.geometry[0]):
        for h_post in range(postPop.geometry[1]):
            # index bounds of the pre-synaptic segment 
            lowerboundX = max(w_post * shiftRF + offsetX,0)
            upperboundX = min(w_post * shiftRF + offsetX + rfX, prePop.geometry[0])
            lowerboundY = max(h_post * shiftRF + offsetY,0)
            upperboundY = min(h_post * shiftRF + offsetY + rfY, prePop.geometry[1])

            for d_post in range(postPop.geometry[2]):
                post_rank = postPop.rank_from_coordinates((w_post,h_post,d_post))
                pre_ranks = []
                weights = []

                for w_pre in range(lowerboundX, upperboundX):
                    for h_pre in range(lowerboundY, upperboundY):
                        for d_pre in range(prePop.geometry[2]):
                            
                            if not ( (prePop == postPop) and (w_post == w_pre) and (h_post == h_pre) and (d_post == d_pre) ): # exclude self connections
                                pre_ranks.append(prePop.rank_from_coordinates((w_pre,h_pre,d_pre)))
                                weights.append(np.random.uniform(min_value, max_value))
                                
                synapses.add(post_rank, pre_ranks, weights, [ delay for i in range(len(pre_ranks))])
    
    return synapses

def patch_pattern2(prePop, postPop, min_value, max_value, shiftRF, rfX, rfY, offsetX, offsetY, delay=0):

    synapses = CSR()
    for w_post in range(postPop.geometry[0]):
        for h_post in range(postPop.geometry[1]):
            # index bounds of the pre-synaptic segment 
            lowerboundX = max(w_post * shiftRF + offsetX,0)
            upperboundX = min(w_post * shiftRF + offsetX + rfX, prePop.geometry[0])
            lowerboundY = max(h_post * shiftRF + offsetY,0)
            upperboundY = min(h_post * shiftRF + offsetY + rfY, prePop.geometry[1])

            # if the "real" upperbound behind the geometry bounds of the prePop, shift the lowerbound to the pre neurons 'behind' you
            if ((w_post * shiftRF + offsetX + rfX) > prePop.geometry[0]) :
                offset = (w_post * shiftRF + offsetX + rfX) - prePop.geometry[0] 
                lowerboundX -= offset

            if ((h_post * shiftRF + offsetY + rfY) > prePop.geometry[1]) :
                offset = (h_post * shiftRF + offsetY + rfY) - prePop.geometry[1] 
                lowerboundY -= offset
                #print(lowerboundY)
            # if the "real" lowerbound smaller zero, shift the upperbound to the pre neurons 'before' you
            if ((w_post * shiftRF + offsetX) < 0) :
                offset = np.abs((w_post * shiftRF + offsetX) - 0) 
                upperboundX += offset
                
            if ((h_post * shiftRF + offsetY) < 0) :
                offset = np.abs((h_post * shiftRF + offsetY) - 0 ) 
                upperboundY += offset

            for d_post in range(postPop.geometry[2]):
                post_rank = postPop.rank_from_coordinates((w_post,h_post,d_post))
                pre_ranks = []
                weights = []

                for w_pre in range(lowerboundX, upperboundX):
                    for h_pre in range(lowerboundY, upperboundY):
                        for d_pre in range(prePop.geometry[2]):
                            
                            if not ( (prePop == postPop) and (w_post == w_pre) and (h_post == h_pre) and (d_post == d_pre) ): # exclude self connections
                                pre_ranks.append(prePop.rank_from_coordinates((w_pre,h_pre,d_pre)))
                                weights.append(np.random.uniform(min_value, max_value))

                synapses.add(post_rank, pre_ranks, weights, [ delay for i in range(len(pre_ranks))])
        
    return synapses
