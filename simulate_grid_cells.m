 clc; clear all;
%%%%	Homogeneous network with integrator neurons		%%%%

% this version includes:
% 1. buidling the network model 
% 2. Running the network model
% 3. Recording the grid cell temporal activity  

% Based on Burak and Fiete 2009's continuous attractor model
% -clean version
% 
% STEPS:
% 1) to generate or load a connectivity matrix based on mexican hat
% connectivity
% 2) initial figure showing the population activity of the network along
% with spatial firing map and normalized rate map
% 3) couplig of rate movement to attractor activity bump and slow drifiting
% movement of activity bump over the entire network 
% 
% PS: At any point for closing the simulation use "Ctrl" + "C".

% Ecah neuron in the recurrent network is an abstract neuron (rate based) and its firing
% rate is governed by synaptic input and synaptic currents are governed by
% single exponential fucntion (single time constant).
%
% Each neuron in the network is sends very strong inhibitory inputs to
% certain set of neurons (ring of inhibition) which is governed by Mexican
% Hat connectivity.
% Each neuron apart from inhibitory inputs receive:
% 1) a constant global input(if inhibition is absent, sufficient to maintain spontaneous activity)
% 2) time-varying velocity dependent current input to each cell.
% 
% Each cell in the network receive a preferred head direction inputs
% To make is scalable the, the network is divided up into 2-by-2 blocks and each cell in a block is
% assigned to one direction N, S, E, or W.
% N cell will recieve more input when animal is moving in north direction
% but lesser in other direction and the input will be proportional velocity
% of the rat movement
% 
% The synaptic output from any particular cell produces a ring of
% inhibition on other cells in the sheet of cells that is slightly offset
% in the direction of the cell's preference (for allowing movement of activity bump on network).
% That is, an N cell inhibits a ring of cells that is centered slightly "north" of the cell. 
% 
% With the ring of inhibition shifted slightly forward, the cell will
% slightly inhibit itself and cells near it, but cells in the non-inhibited
% space ahead in the center of the ring will increase in activity. In this 
% way, making "north" cells active causes the set of active cells to shift 
% slightly along the sheet of cells in some particular direction. This slow
% shifting is what make movement of attractor.
% 
% When the animal is not moving, all of the directional cells get the same
% input and cancel out in their effects.
%
%
% 

livePlot = 0; % update period in steps (for current simulation in ms)

usePeriodicNetwork = 1; %0= aperiodic (decaying envelope), 1 = periodic (torus)

useRealTrajectory = 1; % 0= constant velocity (only for emergence of attractor), 1=load trajectory from data folder 
constantVelocity = 0*[0.5; 0]; % m/s

useCurrentW = 0; % if W exists in data folder, instead of generating new
loadWIfPossible = 0;
saveW = 0; % save W to data folder

%% Cell parameters
% Changing parameters: lambda, beta, alpha
%lambda = 13, 15
%beta = 3/lambda^2, 1.5/lambda^2
%alpha = 50,45
tau = 10; % grid cell synapse time constant, ms
if ~useRealTrajectory
  alpha = 50; % input gain
else
  alpha = 45; %0.10315; % input gain % tuned for my virtual trajectory
end

%% Network/Weight matrix parameters
ncells = 60*60; % size of network
a = 1 ; % if >1, uses local excitatory connections
lambda = 15; % approx the periodicity of the lattice on the neural sheet
beta = 3/(lambda^2);
% This sets how much wider the inhibitory region is than the excitatory
% region and must be large enough that it prevents the population activity
% from merging into one large bump of activity. 
gamma = 1.1*beta;       %1.1

% threshold for plotting a cell as having spiked
spikeThresh = 0.1;

%% Simulation parameters
dt = 1; % time step, ms
simdur = 30000; %200e3; % total simulation time, in ms
stabilizationTime = 100; % no-velocity time for pattern to form, ms
tind = 0; % time step number for indexing
t = 0; % simulation time variable, ms

%% Initial conditions
rng(4)
s = rand(1,ncells); % activation of each cell

%% Firing field plot variables
watchCell = (ncells/2-sqrt(ncells)/2)+5; % specific cell's spatial activity will be plotted
nSpatialBins = 60; % for visualization while running the simulations, for analysis 100
minx = 0; maxx = 2; % m
miny = 0; maxy = 2; % m
occupancy = zeros(nSpatialBins);
spikes = zeros(nSpatialBins);
spikeCoords = [];


%% Create 2-by-ncells preferred direction vector (radians)
dirs = [0 pi/2; pi 3*pi/2];
dirs = repmat(dirs,sqrt(ncells)/2,sqrt(ncells)/2);
dirs = reshape(dirs,1,[]);
dirVects = [cos(dirs); sin(dirs)];

%% Make x a 2-by-ncells vector of the 2D cell positions on the neural sheet
x = (0:(sqrt(ncells)-1))-(sqrt(ncells)-1)/2;
[X,Y] = meshgrid(x,x);
x = [reshape(X,1,[]); reshape(Y,1,[])];
cellSpacing = Y(2)-Y(1); % sets length of field shift in recurrent connections
ell = 2*cellSpacing; % offset of center of inhibitory output
cellDists = sqrt(x(1,:).^2 + x(2,:).^2); % distance from (0,0) for A below

%% Weight matrix
% simple function for Mexican Hat connectivity can also be used instead
wSparseThresh = -1e-6; -1e-8;
if ~(useCurrentW && exist('W','var'))
  if usePeriodicNetwork
    fname = sprintf('data/W_Bu09_torus_n%d_l%1g_a%d_alpha%g_lambda%g.mat',ncells,ell,a,alpha,lambda);
  else
    fname = sprintf('data/W_Bu09_aperiodic_n%d_l%1g.mat',ncells,ell);
  end
  if loadWIfPossible && exist(fname,'file')
    fprintf('Attempting to load pre-generated W...\n')
    load(fname);
    fprintf('+ Loaded pre-generated W. Using a = %2g, lambda = %2g, beta = %2g, gamma = %2g, ell = %d\n',a,lambda,beta,gamma,ell)
  else
    fprintf('Generating new W. This may take a while. Notifications at 10%% intervals...\n')

    % Define inputs weights for each neuron i one at a time
    W = [];
    for i=1:ncells
      if mod(i,round(ncells/10))==0
        fprintf('Generating weight matrix. %d%% done.\n',round(i/ncells*100))
      end
      if usePeriodicNetwork
        clear squaredShiftLengths;
        % Guanella et al 2007's approach to the periodic distance function
        shifts = repmat(x(:,i),1,ncells) - x - ell*dirVects;
        squaredShiftLengths(1,:) = shifts(1,:).^2 + shifts(2,:).^2;
        shifts = repmat(x(:,i),1,ncells) - x - sqrt(ncells)*[ones(1,ncells); zeros(1,ncells)] - ell*dirVects;
        squaredShiftLengths(2,:) = shifts(1,:).^2 + shifts(2,:).^2;
        shifts = repmat(x(:,i),1,ncells) - x + sqrt(ncells)*[ones(1,ncells); zeros(1,ncells)] - ell*dirVects;
        squaredShiftLengths(3,:) = shifts(1,:).^2 + shifts(2,:).^2;
        shifts = repmat(x(:,i),1,ncells) - x - sqrt(ncells)*[zeros(1,ncells); ones(1,ncells)] - ell*dirVects;
        squaredShiftLengths(4,:) = shifts(1,:).^2 + shifts(2,:).^2;
        shifts = repmat(x(:,i),1,ncells) - x + sqrt(ncells)*[zeros(1,ncells); ones(1,ncells)] - ell*dirVects;
        squaredShiftLengths(5,:) = shifts(1,:).^2 + shifts(2,:).^2;
        shifts = repmat(x(:,i),1,ncells) - x + sqrt(ncells)*[ones(1,ncells); ones(1,ncells)] - ell*dirVects;
        squaredShiftLengths(6,:) = shifts(1,:).^2 + shifts(2,:).^2;
        shifts = repmat(x(:,i),1,ncells) - x + sqrt(ncells)*[-1*ones(1,ncells); ones(1,ncells)] - ell*dirVects;
        squaredShiftLengths(7,:) = shifts(1,:).^2 + shifts(2,:).^2;
        shifts = repmat(x(:,i),1,ncells) - x + sqrt(ncells)*[ones(1,ncells); -1*ones(1,ncells)] - ell*dirVects;
        squaredShiftLengths(8,:) = shifts(1,:).^2 + shifts(2,:).^2;
        shifts = repmat(x(:,i),1,ncells) - x + sqrt(ncells)*[-1*ones(1,ncells); -1*ones(1,ncells)] - ell*dirVects;
        squaredShiftLengths(9,:) = shifts(1,:).^2 + shifts(2,:).^2;
        
        % Select respective least distances:
        squaredShiftLengths = min(squaredShiftLengths);
      else
        shifts = repmat(x(:,i),1,ncells) - x - ell*dirVects;
        squaredShiftLengths = shifts(1,:).^2 + shifts(2,:).^2;
      end
      temp = a*exp(-gamma*squaredShiftLengths) - exp(-beta*squaredShiftLengths);
      temp(temp>wSparseThresh) = 0;
      W = [W; sparse(temp)];
    end

    if saveW
      save(fname,'W','a','lambda','beta','gamma','ell','-v7.3');
    end
  end
end

%% Define envelope function
if usePeriodicNetwork
  % Periodic
  A = ones(size(cellDists));
else
  % Aperiodic
  R = sqrt(ncells)/2; % radius of main network in cell-position units
  a0 = sqrt(ncells)/32; % envelope fall-off rate
  dr = sqrt(ncells)/2; % diameter of non-tapered region, in cell-position units
  A = exp(-a0*(((cellDists)-R+dr)/dr).^2);
  nonTaperedInds = find(cellDists < (R-dr));
  A(nonTaperedInds) = 1;  
end

%% Make optional figure of sheet of activity
if livePlot
  h = figure('color','w','name','Activity of sheet of cells on brain''s surface');
  drawnow
end

%% Possibly load trajectory from disk
if useRealTrajectory
    load 'Codes/data/pos_2m.mat'
  %load /Users/divyansh/lab_work/Network_model/Grid_hetero_res_full_run/Homo_int/data/pos_2m.mat;
  % use below code if you are using real Hafting 2005 trajectory
  %our time units are in ms so:
%   pos(3,:) = pos(3,:)*1e3;
  % interpolate down to simulation time step
%   pos = [interp1(pos(3,:),pos(1,:),0:dt:pos(3,end));
%          interp1(pos(3,:),pos(2,:),0:dt:pos(3,end));
%          interp1(pos(3,:),pos(3,:),0:dt:pos(3,end))];
%   pos(1:2,:) = pos(1:2,:)/100; % cm to m
  vels = [diff(pos(1,:)); diff(pos(2,:))]/dt; % m/s
end

tic
%% Simulation
fprintf('Simulation starting. Press ctrl+c to end...\n')
while t<simdur-1
  tind = tind+1;
  t = dt*tind;
  
  % Velocity input
  if t<stabilizationTime
    v = [0; 0]; % m/s
  else
    if ~useRealTrajectory
      v = constantVelocity; % m/s
    else
      v = vels(:,tind); % m/s
    end
  end
  curDir(tind) = atan2(v(2),v(1)); % rad
  speed(tind) = sqrt(v(1)^2+v(2)^2);%/dt; % m/s

  
  % Feedforward input
  B = A.*(1+alpha*dirVects'*v)';
  
  % Total synaptic driving currents
  sInputs = (1*W*s')' + B;

  % Synaptic drive only increases if input cells are over threshold 0
  sInputs = sInputs.*(sInputs>0);

  % Synaptic drive decreases with time constant tau
  s = s + dt*(sInputs - s)/tau;
  
  
  rcd(:,t) = s(:);
  
  % Save firing field information (average value of s in each spatial bin)
  if useRealTrajectory
    if s(watchCell)>spikeThresh
      spikeCoords = [spikeCoords; pos(1,tind) pos(2,tind)];
    end
    xindex = round((pos(1,tind)-minx)/(maxx-minx)*nSpatialBins);
    yindex = round((pos(2,tind)-miny)/(maxy-miny)*nSpatialBins);
    if (xindex>60)
        xindex = 60;
    end
    if (yindex>60)
        yindex = 60;
    end
    if (xindex<2)
        xindex = 2;
    end
    if (yindex<2)
        yindex = 2;
    end
    occupancy(yindex,xindex) = occupancy(yindex,xindex) + dt;
    spikes(yindex,xindex) = spikes(yindex,xindex) + s(watchCell);
  end

  if livePlot>0 && (livePlot==1 || mod(tind,livePlot)==1)
    if ~useRealTrajectory
      figure(h);
      set(h,'name','Activity of sheet of cells on brain''s surface');
      imagesc(reshape(s,sqrt(ncells),sqrt(ncells)));
      axis square
      set(gca,'ydir','normal')
      title(sprintf('t = %.1f ms',t))
      drawnow
      
      
    else
      figure(h);
      title(fname)
      subplot(131);
      imagesc(reshape(s,sqrt(ncells),sqrt(ncells)));
      axis square
      title('Population activity')
      set(gca,'ydir','normal')
      subplot(132);
      imagesc(spikes./occupancy);
      axis square
      set(gca,'ydir','normal')
      title({sprintf('t = %.1f ms',t),'Rate map'})
      subplot(133);
      plot(pos(1,1:tind),pos(2,1:tind));
      hold on;
      if ~isempty(spikeCoords)
        plot(spikeCoords(:,1),spikeCoords(:,2),'r.')
      end
      axis square
      title({'Trajectory (blue)','and spikes (red)'})
      drawnow
    end
  end
end
toc

% %% To plot
% 
% % Population activity
% rate_map = reshape(rcd(:,29999),sqrt(ncells),sqrt(ncells));
% figure; imagesc(rate_map)
% 
% % Trajectory
% for tind = 299999 %(last timestep)
%     plot(pos(1,1:tind),pos(2,1:tind));
%     hold on;
%     if ~isempty(spikeCoords)
%        plot(spikeCoords(:,1),spike_coords_13(:,2),'r.')
%     end
%     axis square
% end

%% Change parameters, save results accordingly
rcd_lambda_15 = rcd;
pos_15 = pos;
spike_coords_15 = spikeCoords;
save("results_lambda_15.mat", 'rcd_lambda_15', 'pos_15', 'spike_coords_15', '-v7.3')
