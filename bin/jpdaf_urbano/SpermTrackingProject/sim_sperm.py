# Generated with SMOP  0.41
from libsmop import *
# sim_sperm.m

    
@function
def sim_sperm(avg_speed=None,t_sim=None,T=None,tau=None,wob_flag=None,*args,**kwargs):
    varargin = sim_sperm.varargin
    nargin = sim_sperm.nargin

    # Sperm parameters
    Vmag=dot(avg_speed,(dot(0.5,randn) + 1))
# sim_sperm.m:4
    # Swimming parameters
    phi=dot(dot(2,pi),rand)
# sim_sperm.m:7
    
    alh=5 + randn
# sim_sperm.m:8
    
    bcf=5 + randn
# sim_sperm.m:9
    
    x=0
# sim_sperm.m:11
    y=0
# sim_sperm.m:12
    # Simulate particle motion
    for k in arange(1,(t_sim / T)).reshape(-1):
        vx=dot(Vmag,cos(phi))
# sim_sperm.m:18
        vy=dot(Vmag,sin(phi))
# sim_sperm.m:19
        phi_dot=dot(sqrt(2 / tau),randn)
# sim_sperm.m:22
        x=x + dot(vx,T)
# sim_sperm.m:25
        y=y + dot(vy,T)
# sim_sperm.m:26
        phi=phi + dot(phi_dot,T)
# sim_sperm.m:29
        wob=dot(alh,cos(dot(dot(dot(dot(2,pi),bcf),k),T)))
# sim_sperm.m:32
        uvx=vx / Vmag
# sim_sperm.m:33
        uvy=vy / Vmag
# sim_sperm.m:34
        ut=dot(concat([[0,- 1],[1,0]]),concat([[uvx],[uvy]]))
# sim_sperm.m:35
        wx=dot(wob,ut(1)) + randn
# sim_sperm.m:36
        wy=dot(wob,ut(2)) + randn
# sim_sperm.m:37
        hx=x + dot(wx,wob_flag)
# sim_sperm.m:39
        hy=y + dot(wy,wob_flag)
# sim_sperm.m:40
        Z[1,k]=hx
# sim_sperm.m:43
        Z[2,k]=hy
# sim_sperm.m:44
    