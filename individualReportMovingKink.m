%% ==================== Sidney Sweeney Kink Dynamics Individual Extension ===================
%% ====================================== Moving Kink =======================================

%% ======================================= Grid Setup =======================================
clc; clear; close all;
% Number of gridpoints and domain size
N = 1000; L = 100;

% Spatial Grid
x = linspace(-L,L,N)';
dx = x(2)-x(1);

%% ===================================== Initialisation =====================================
% Inital "guess"
speed = 0.3; x0 = -10; gamma = 1/sqrt(1-speed^2);
z = gamma*(x - x0); % kink at x0=-10 moving to the right
phi0 = 4*atan(exp(z));
v0  = - (gamma/2) * sech(z);

%% =============================== Potential and Derivatives ================================
U = @(phi) 1-cos(phi); % Sine-Gordon potential
U_p = @(phi) sin(phi);
U_pp = @(phi) cos(phi);

%% ================================== Time Parameter Setup ==================================
T = 100; dt = 0.001; time = 0:dt:T;
tspan = [0,T];

% Initial State
y0 = [phi0; v0];

%% ==================================== Laplacian matrix ====================================
Laplacian = build_laplacian(N, dx); % Neumann boundary conditions
function Lap = build_laplacian(N, dx)
    one_vec = ones(N,1);
    Lap = spdiags([one_vec, -2*one_vec, one_vec], -1:1, N, N);
    Lap(1,:) = 0; Lap(1,1) = -2; Lap(1,2) = 2;
    Lap(end,:) = 0; Lap(end,end) = -2; Lap(end,end-1) = 2;
    Lap = (1/dx^2)*Lap;
end

%% ===================================== ODE113 Solver =====================================
function dydt = kink_rhs(y, Lap, U_p)
    N = numel(y)/2;
    phi = y(1:N);
    v   = y(N+1:end);

    phi_t = v;

    Lap_phi = Lap * phi;
    v_t = Lap_phi - U_p(phi);
    dydt = [phi_t; v_t];
end

[T, Y] = ode113(@(t,y) kink_rhs(y,Laplacian,U_p), tspan, y0);
state = Y(:, 1:N)';

%% ============================ Plots at Equidistant Timestamps =============================
plot(x, state(:,1),'-*');hold on;
plot(x, state(:,round(end/4)),'-*')   % T/4
plot(x, state(:,round(end/2)),'-*')   % T/2
plot(x, state(:,round(3*end/4)),'-*') % 3T/4
plot(x, state(:, end),'-*');
xlabel('x', 'Interpreter','latex');
ylabel('$\phi$', 'Interpreter','latex');
set(gca,'fontsize',20);
title('\phi(x,t) at Equidistant Timestamps');
grid on;
legend('$\phi(x,t=0)$','$\phi(x,t=T/4)$','$\phi(x,t=T/2)$','$\phi(x,t=3T/4)$', '$\phi(x,t=T)$','Interpreter','latex', 'Location','best')
[~, idx0] = min(abs(x));

%% ======================================= Animations =======================================
% Save animation as MP4 (using ode113 solver)
filename = append('moving_kink_animation.mp4');
video = VideoWriter(filename, 'MPEG-4');
video.FrameRate = 10; %video.Quality = 100;
open(video);

step = max(1, floor(length(T) / 200));
figure;

for n = 1:step:length(T)
        plot(x, state(:,n), 'LineWidth', 2);
        hold on;
        ylim([-2 8]);
        xlim([-20 20]);
        set(gca,'fontsize',16);
        xlabel('x', 'interpreter','latex'); ylabel('$\phi$', 'interpreter','latex');
        title(sprintf('\\phi(x,t) with initial speed v = %.3f,  t = %.3f', speed, T(n)));
        grid on;
        legend('$\phi(x, t)$', 'Interpreter','latex'); 
        hold off;
        frame = getframe(gcf);
        writeVideo(video, frame);
end
close(video);
disp(append('Saved MP4 animation as ', filename));

