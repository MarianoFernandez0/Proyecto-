function [Z, Vmag, alh, bcf] = sim_sperm(avg_speed, t_sim, T, tau, wob_flag)

% Sperm parameters
Vmag = avg_speed * (0.5 * randn + 1);

% Swimming parameters 
phi  = 2 * pi * rand;   % (rad) initial heading angle
alh  = 5 + randn;       % (um) amplitude lateral head displacement
bcf  = 5 + randn;       % (hz) beat cross frequency

x = 0;
y = 0;

% Simulate particle motion
for k = 1:(t_sim/T);    
    
    % Velocity vector (um/sec)
    vx = Vmag * cos(phi);
    vy = Vmag * sin(phi);
    
    % Swimming angle rate (rad/sec)
    phi_dot = sqrt(2/tau) * randn;
                
    % Position vector (um)
    x = x + vx * T;
    y = y + vy * T;
    
    % Swimming angle (rad)
    phi = phi + phi_dot * T;
       
    % Head position vector (um)
    wob = alh * cos( 2 * pi * bcf * k * T);
    uvx = vx / Vmag;
    uvy = vy / Vmag;
    ut = [0 -1; 1 0] * [uvx; uvy];
    wx = wob * ut(1) + randn;
    wy = wob * ut(2) + randn;
    
    hx = x + wx * wob_flag;
    hy = y + wy * wob_flag;
        
    % Record measured position
    Z(1,k) = hx;
    Z(2,k) = hy;           
    
end