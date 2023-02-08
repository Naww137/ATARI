
addpath('/Users/noahwalton/software/PSwarmM_v2_1')

tStart = tic ; 

%% import data
case_file = './syndat_data_SLBW.hdf5';
isample = 3 ;

% Load data as a table
exp_pw = read_hdf5(case_file, sprintf('/sample_%i/exp_pw', isample)) ;
exp_cov = h5read(case_file, sprintf('/sample_%i/exp_cov', isample));
theo_par = read_hdf5(case_file, sprintf('/sample_%i/theo_par', isample)) ; 
theo_chi2 = (exp_pw.theo_trans-exp_pw.exp_trans)' * inv(exp_cov) *  (exp_pw.theo_trans-exp_pw.exp_trans) ;

% Nuclear parameters
A=180.948434; 
Constant=0.00219680781008; %sqrt(2Mn)/hbar; units of (10^(-12) cm sqrt(eV)^-1)
Ac=0.81271; % scattering radius 6.7 fermi expressed as 10^-12 cm
I=3.5; % target angular Momentum
ii=0.5; % incident angular momentum
l=0;   % l=0 or s-wave spin group

s=I-ii; %
J=l+s;  % 
g=(2*J+1)/( (2*ii+1)*(2*I+1) );   % spin statistical factor g sub(j alpha)

pig=pi*g;

% Functions
k=@(E) Constant*(A/(A+1))*sqrt(E);   % wave number
rho=@(E) k(E)*Ac;       % related to the center of mass momentum
P=@(E) rho(E);          % penatrability factor

%% solution vector

% Extract resonance ladder values from Syndat
Elevels = theo_par.E';
Gc = 0.001 * theo_par.Gg';
gn_square = 0.001 * theo_par.gnx2';


% Number of resonances
num_res_actual = height(theo_par);
NumPeaks = 6; % Number of resonance guesses

% Baron formatted vector of solution parameters
% Last num_res_guess values are binary switches for whether or not resonance exists
sol_w = zeros(1, 4 * NumPeaks);
for j=1:num_res_actual
    sol_w(3*(j-1)+1) = Gc(j);
    sol_w(3*(j-1)+2) = gn_square(j);
    sol_w(3*(j-1)+3) = Elevels(j);
    sol_w(3 * NumPeaks + j) = 1;
end


% Define energy grid
WE = exp_pw.E';  % Passing Energies to WE is artifact of Jordan-Noah edits

% Create function pieces
Pen = P(WE);
phi = rho(WE);
kE = k(WE);

% Create total cross section function
xs_func = @(w) 0;
for jj=1:NumPeaks % here index 1 is Gc, 2 is gn, 3 is Elevel
%     
    xs_func = @(w) xs_func(w) + (w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen).*w(2+3*(jj-1)).*2.*Pen./4./((w(3+3*(jj-1))-WE).^2 + ((w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen)./2).^2) ...
                                             .*cos(2.*phi) ...
                                         - (w(3+3*(jj-1))-WE).*(w(2+3*(jj-1)).*2.*Pen)./2./( (w(3+3*(jj-1))-WE).^2 + (( w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen )./2).^2 ) ...
                                             .*sin(2.*phi)   ;
end
xs_func = @(w) (4.*pig./kE.^2) .* (sin(phi).^2+xs_func(w));

n = 0.067166; 
trans_func = @(w) exp(-n*xs_func(w));

WC = exp_pw.exp_trans';

% Plot theoretical and experimental syndat data
figure(1); clf
errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
plot(WE, trans_func(sol_w),'o', 'DisplayName', 'Matlab theo') ; hold on
plot(WE, exp_pw.theo_trans, 'DisplayName', 'Syndat theo')
legend()

%% setup optimization and constraints

fobj.ObjFunction =@(w) (trans_func(w)-WC) * inv(exp_cov) *  (trans_func(w)-WC)' ;
% fobj.ObjFunction =@(w) sum((trans_func(w)-WC).^2);


% insert min/max of Gc, gn_square, and energy 
MinVec = [0 0 min(WE)];
MaxVec = [max(Gc)*5 max(gn_square)*5 max(WE)];

RM_PerPeak = 3 ;

TotalRM_PerWindow = NumPeaks*RM_PerPeak;
TotalParm_PerWindow=NumPeaks*(RM_PerPeak);

minimum = []; maximum = [];
for jj=1:NumPeaks
    minimum = [minimum; MinVec'];
    maximum = [maximum; MaxVec'];
end

fobj.Variables = TotalParm_PerWindow ;
fobj.LB = minimum;
fobj.UB = maximum;

InitialPopulation = [];
% InitialPopulation(1).x = [];
% InitialPopulation(2).x = []; second guess

opt=PSwarm('defaults') ;
opt.IPrint = 100; 
opt.MaxObj = 1e5*30;
opt.MaxIter = 1e10 ; %opt.MaxObj ;
opt.CPTolerance = 1e-5; %1e-7;
opt.DegTolerance = 1e-6;


%%
% domain = linspace(fobj.LB(1),fobj.UB(1),100);
% figure('WindowStyle','docked')
% plot(domain,fobj(domain))


%%
[BestParticle, BestParticleObj, RunData] = PSwarm(fobj,InitialPopulation,opt);

%%

figure(1); clf
errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
plot(WE, trans_func(sol_w), 'DisplayName', 'Matlab theo') ; hold on
% plot(WE, exp_pw.theo_trans, 'DisplayName', 'Syndat theo'); hold on
plot(WE, trans_func(BestParticle), 'DisplayName','PSwarm sol');
legend()

fprintf('SE solution: %f\n',fobj.ObjFunction(sol_w))
fprintf('SE PSwarm: %f\n', BestParticleObj)


