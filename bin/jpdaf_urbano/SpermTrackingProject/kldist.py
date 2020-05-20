# Generated with SMOP  0.41
from libsmop import *
# kldist.m

    
@function
def kldist(XI=None,XJ=None,*args,**kwargs):
    varargin = kldist.varargin
    nargin = kldist.nargin

    # Implementation of the Kullback-Leibler Divergence to use with pdist
  # (cf. "The Earth Movers' Distance as a Metric for Image Retrieval",
  #      Y. Rubner, C. Tomasi, L.J. Guibas, 2000)
    
    # @author: B. Schauerte
  # @date:   2009
  # @url:    http://cvhci.anthropomatik.kit.edu/~bschauer/
    
    # Copyright 2009 B. Schauerte. All rights reserved.
  # 
  # Redistribution and use in source and binary forms, with or without 
  # modification, are permitted provided that the following conditions are 
  # met:
  # 
  #    1. Redistributions of source code must retain the above copyright 
  #       notice, this list of conditions and the following disclaimer.
  # 
  #    2. Redistributions in binary form must reproduce the above copyright 
  #       notice, this list of conditions and the following disclaimer in 
  #       the documentation and/or other materials provided with the 
  #       distribution.
  # 
  # THIS SOFTWARE IS PROVIDED BY B. SCHAUERTE ''AS IS'' AND ANY EXPRESS OR 
  # IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
  # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
  # DISCLAIMED. IN NO EVENT SHALL B. SCHAUERTE OR CONTRIBUTORS BE LIABLE 
  # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
  # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
  # BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
  # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  # OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  # ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  # 
  # The views and conclusions contained in the software and documentation
  # are those of the authors and should not be interpreted as representing 
  # official policies, either expressed or implied, of B. Schauerte.
    
    m=size(XJ,1)
# kldist.m:40
    
    p=size(XI,2)
# kldist.m:41
    
    
    assert_(p == size(XJ,2))
    
    assert_(size(XI,1) == 1)
    
    
    d=zeros(m,1)
# kldist.m:46
    
    
    for i in arange(1,m).reshape(-1):
        for j in arange(1,p).reshape(-1):
            #d(i,1) = d(i,1) + (XJ(i,j) * log(XJ(i,j) / XI(1,j))); # XI is the model!
            if XI(1,j) != 0:
                d[i,1]=d(i,1) + (dot(XI(1,j),log(XI(1,j) / XJ(i,j))))
# kldist.m:52
    