
tStart = tic ; 

%% import data
case_file = '../perf_test_staticladder.hdf5';
isample = 3 ;

% Load data as a table
exp_pw = read_hdf5(case_file, sprintf('/sample_%i/exp_pw', isample)) ;
exp_cov = h5read(case_file, sprintf('/sample_%i/exp_cov', isample));
theo_par = read_hdf5(case_file, sprintf('/sample_%i/theo_par', isample)) ; 
theo_chi2 = (exp_pw.theo_trans-exp_pw.exp_trans)' * inv(exp_cov) *  (exp_pw.theo_trans-exp_pw.exp_trans) ;
theo_likelihood = chi2pdf(theo_chi2, length(exp_pw.theo_trans));

%% Nuclear Parameters and functions for scattering theory 

A=180.948434; %Cu-63, number from cu63 input txt file
Constant=0.00219680781008; %sqrt(2Mn)/hbar; units of (10^(-12) cm sqrt(eV)^-1)
Ac=0.81271; % scattering radius 6.7 fermi expressed as 10^-12 cm
I=3.5; % target angular Momentum
ii=0.5; % incident angular momentum
l=0;   % l=0 or s-wave spin group

s=I-ii; %
J=l+s;  % 
g=(2*J+1)/( (2*ii+1)*(2*I+1) );   % spin statistical factor g sub(j alpha)

pig=pi*g;

%l=0   or s-wave spin group energy dependent functions of the wave number
k=@(E) Constant*(A/(A+1))*sqrt(E);   % wave number
rho=@(E) k(E)*Ac;       % related to the center of mass momentum
P=@(E) rho(E);          % penatrability factor
% phi = @(E) rho(E) ; 


% Extract resonance ladder values from Syndat
Elevels = theo_par.E';
Gc = 0.001 * theo_par.Gg';
gn_square = 0.001 * theo_par.gnx2';


% Number of resonances
num_res_actual = height(theo_par);
NumPeaks = num_res_actual; % Number of resonance guesses

% obj function formated solution vector
sol_w = zeros(1, 4 * NumPeaks);
for j=1:num_res_actual
    sol_w(3*(j-1)+1) = Gc(j);
    sol_w(3*(j-1)+2) = gn_square(j);
    sol_w(3*(j-1)+3) = Elevels(j);
    sol_w(3*NumPeaks+j) = 1;
end


% Define energy grid
WE = exp_pw.E';  % Passing Energies to WE is artifact of Jordan-Noah edits


% create cross section function(s)
% total
Pen = P(WE);
phi = rho(WE);
kE = k(WE) ;
xs_func = @(w) 0;
for jj=1:NumPeaks % here index 1 is Gc, 2 is gn, 3 is Elevel
    
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


%% solve baron 1 Original implementation - Works!

% fun_robust1=@(w) sum((xs_func(w)-WC).^2);
fun_robust1=@(w) sum((trans_func(w)-WC).^2);

% insert min/max of Gc, gn_square, and energy 
MinVec = [0 0 min(WE)];
MaxVec = [25 10 max(WE)];

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
 
% EnergyOrder=zeros(NumPeaks-1,4*NumPeaks);
% PeakSpacing=1;
% for jj=1:(NumPeaks-1)
%     EnergyOrder(jj,RM_PerPeak+RM_PerPeak*(jj-1))=-1;
%     EnergyOrder(jj,RM_PerPeak+RM_PerPeak*jj)=1;
%     EnergyOrder(jj,TotalRM_PerWindow+jj)=-PeakSpacing/2;
%     EnergyOrder(jj,TotalRM_PerWindow+(jj+1))=-PeakSpacing/2;
% end
% 
% A = [A_Lower;A_Upper;EnergyOrder];
% 
% SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow),zeros(1,NumPeaks-1)];
% SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow),inf(1,NumPeaks-1)];

A = [A_Lower;A_Upper]; 
SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow)];
SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow)];

lb=zeros(1,TotalParm_PerWindow);
ub=[repmat(MaxVec,1,NumPeaks),ones(1,NumPeaks)];

% baron runtime options
Options=baronset('threads',8,'PrLevel',1,'MaxTime',2*60, 'EpsA', fun_robust1(sol_w), 'filekp', 0);

%%
xtype=squeeze(char([repmat(["C","C","C"],1,NumPeaks),repmat(["B"],1,NumPeaks)]))';

w0 = w1; % optional inital guess
[w1,fval,ef,info]=baron(fun_robust1,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,w0, Options); % run baron


%%


figure(1); clf
errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
plot(WE, trans_func(sol_w), 'DisplayName', 'Matlab theo') ; hold on
% plot(WE, exp_pw.theo_trans, 'DisplayName', 'Syndat theo'); hold on
plot(WE, trans_func(w1), 'DisplayName','Baron sol');
legend()


fprintf('SE solution: %f\n',fun_robust1(sol_w))
fprintf('SE Baron: %f\n', fval)
