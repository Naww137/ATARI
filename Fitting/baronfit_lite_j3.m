%% Notes

% CHANGES to baronfit_lite that Noah sent in mid-december:

% This .m has been modified to read in .csv files produced from Syndat
% instead of manually entering isotope parameter values 

% This .m does not produce synthetic data. All of that code has been
% removed.

% CURRENT ISSUES

% Baron does not converge to solution, and it does not always stop at
% MaxTime. It stops when set to 30, otherwise it stops at 1000s. 


%% Read in data

% Clear out previous
clear
clc

% Identify folder with syndat data
folder = "/Users/noahwalton/research_local/resonance_fitting/synthetic_data/Ta_1/";

% ID file number (will use this to loop later)
file_number = num2str(2);

% Create file paths
rl_path = strcat(folder, 'rl_', file_number, '.csv');
syndat_path = strcat(folder, 'syndat_', file_number, '.csv');
particle_path = strcat(folder, 'particle.csv');

% Load data
resonance_ladder = readtable(rl_path);
syndat = readtable(syndat_path);
particle = readtable(particle_path);

%% Data clean
% This section replaces missing cross section values with theoretical +
% naive noise values

% Determine where NaNs are
where_nans = ismissing(syndat.exp_xs_tot);

% Replace NaNs with zeros as intermediate step
no_nans = fillmissing(syndat.exp_xs_tot, 'constant', 0);

% Create noise vector
rng(1) % Set seed
rnorms = normrnd(0,1,[height(syndat),1]);

% Replace NaNs with theoretical + noise
syndat.exp_xs_tot = no_nans + where_nans .* (syndat.theo_xs_tot + rnorms);

% Flip syndat
% syndat = flip(syndat, 1); % Doesn't fix it

%% Nuclear Parameters and functions for scattering theory 

% Input parameters

    % Read from Syndat
    A = particle.M;
    Ac = particle.ac;
    I = particle.I;
    ii = particle.i;

    % Manual input
    Constant=0.00219680781008; %sqrt(2*mass_neutron)/hbar; units of (10^(-12) cm sqrt(eV)^-1)
    l=0;   % l=0 or s-wave spin group

% Computed parameters
s=I-ii; 
J=l+s;  
g=(2*J+1)/( (2*ii+1)*(2*I+1) );   % spin statistical factor g sub(j alpha)
pig=pi*g;

% Functions
k=@(E) Constant*(A/(A+1))*sqrt(E);   % wave number
rho=@(E) k(E)*Ac;       % related to the center of mass momentum
P=@(E) rho(E);          % penatrability factor

%% solution vector

% Extract resonance ladder values from Syndat
Elevels = resonance_ladder.E';
Gc = 0.001 * resonance_ladder.Gg';
gn_square = 0.001 * resonance_ladder.gnx2';


% Number of resonances
num_res_actual = height(resonance_ladder);
NumPeaks = num_res_actual; % Number of resonance guesses

% Baron formatted vector of solution parameters
% Last num_res_guess values are binary switches for whether or not resonance exists
sol_w = zeros(1, 4 * NumPeaks);
for j=1:num_res_actual
    sol_w(3*(j-1)+1) = Gc(j);
    sol_w(3*(j-1)+2) = gn_square(j);
    sol_w(3*(j-1)+3) = Elevels(j);
    sol_w(3 * NumPeaks + j) = 1;
end


%% Create total cross section function

% Define energy grid
Energies = syndat.E';
WE = Energies;  % Passing Energies to WE is artifact of Jordan-Noah edits

% Create function pieces
Pen = P(WE);
phi = rho(WE);
kE = k(WE);

% Create total cross section function
xs_func = @(w) 0;
for jj=1:NumPeaks % here index 1 is Gc, 2 is gn, 3 is Elevel
    
    xs_func = @(w) xs_func(w) + (w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen).*w(2+3*(jj-1)).*2.*Pen./4./((w(3+3*(jj-1))-WE).^2 + ((w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen)./2).^2) ...
                                             .*cos(2.*phi) ...
                                         - (w(3+3*(jj-1))-WE).*(w(2+3*(jj-1)).*2.*Pen)./2./( (w(3+3*(jj-1))-WE).^2 + (( w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen )./2).^2 ) ...
                                             .*sin(2.*phi)   ;
end
xs_func = @(w) (4.*pig./kE.^2) .* (sin(phi).^2+xs_func(w));

% Use syndat data instead of Jordan's noisy data
WC = syndat.exp_xs_tot';

% Plot theoretical and experimental syndat data
figure(1); clf
scatter(Energies, WC, '.', 'DisplayName', 'Syndat exp'); hold on
plot(Energies, xs_func(sol_w), 'DisplayName', 'Matlab theo') ; hold on
plot(Energies, syndat.theo_xs_tot, 'DisplayName', 'Syndat theo')
legend()


%% Set Baron options

fun_robust1=@(w) sum((xs_func(w)-WC).^2);

% insert min/max of Gc, gn_square, and energy 
MinVec = [0 0 min(Energies)];
MaxVec = [25 10 max(Energies)];

% automated setup of baron inputs
RM_PerPeak = 3 ;
TotalRM_PerWindow = NumPeaks*RM_PerPeak;
TotalParm_PerWindow=NumPeaks*(RM_PerPeak+1);

A_Lower=[diag(ones(1,TotalRM_PerWindow)),zeros(TotalRM_PerWindow,NumPeaks)];
A_Upper=[diag(ones(1,TotalRM_PerWindow)),zeros(TotalRM_PerWindow,NumPeaks)];
for jj=1:NumPeaks
    Index1=3*(jj-1); % striding function
    Index2=TotalRM_PerWindow+jj;
    A_Lower([1+Index1,2+Index1,3+Index1],Index2)=-MinVec;
    A_Upper([1+Index1,2+Index1,3+Index1],Index2)=-MaxVec;
end
 
EnergyOrder=zeros(NumPeaks-1,4*NumPeaks);
PeakSpacing=2;
for jj=1:(NumPeaks-1)
    EnergyOrder(jj,RM_PerPeak+RM_PerPeak*(jj-1))=-1;
    EnergyOrder(jj,RM_PerPeak+RM_PerPeak*jj)=1;
    EnergyOrder(jj,TotalRM_PerWindow+jj)=-PeakSpacing/2;
    EnergyOrder(jj,TotalRM_PerWindow+(jj+1))=-PeakSpacing/2;
end

A = [A_Lower;A_Upper;EnergyOrder];

SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow),zeros(1,NumPeaks-1)];
SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow),inf(1,NumPeaks-1)];

lb=zeros(1,TotalParm_PerWindow);
ub=[repmat(MaxVec,1,NumPeaks),ones(1,NumPeaks)];

% baron runtime options
Options=baronset('threads',10,'PrLevel',1,'MaxTime',2*60, 'EpsA', sum((xs_func(sol_w)-WC).^2));
xtype=squeeze(char([repmat(["C","C","C"],1,NumPeaks),repmat(["B"],1,NumPeaks)]))';

w0 = [sol_w*1.01]; % optional inital guess


%% Baron solve

[w1,fval,ef,info]=baron(fun_robust1,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,w0, Options); % run baron

%% Plot results

figure(1); clf
scatter(Energies, WC, '.', 'DisplayName', 'Syndat exp'); hold on
plot(Energies, xs_func(sol_w), 'DisplayName', 'Matlab theo') ; hold on
plot(Energies, syndat.theo_xs_tot, 'DisplayName', 'Syndat theo'); hold on
plot(Energies, xs_func(sol_w*1.01), 'DisplayName','Baron sol');
legend()

fprintf('SE solution: %f\n',fun_robust1(sol_w))
fprintf('SE Baron: %f\n', fval)
