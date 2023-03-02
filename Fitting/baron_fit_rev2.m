function baron_fit_rev2(case_file, isample)

% Description
% same as baron_fit_rev1 but not using mixed integer or semi-continuous
% constraints, rather, the widths are allowed to go to zero.

tStart = tic ; 
initial_guess = true;
plotting = false ;

% case_file = './perf_test_staticladder.hdf5';
% isample = 0 ;

% Load data as a table
exp_pw = read_hdf5(case_file, sprintf('/sample_%i/exp_pw', isample)) ;
exp_cov = h5read(case_file, sprintf('/sample_%i/exp_cov', isample));
theo_par = read_hdf5(case_file, sprintf('/sample_%i/theo_par', isample)) ; 
% theo_chi2 = (exp_pw.theo_trans-exp_pw.exp_trans)' * inv(exp_cov) *  (exp_pw.theo_trans-exp_pw.exp_trans) ;
% theo_SE = sum((exp_pw.theo_trans-exp_pw.exp_trans).^2);


if initial_guess
%     suggested_peaks = readmatrix(sprintf('./suggested_peaks_%i.csv', isample)); 
    suggested_peaks = read_hdf5(case_file, sprintf('/sample_%i/poles', isample)) ;
    suggested_peaks = suggested_peaks.E; 
end

%% Nuclear Parameters and functions for scattering theory 

% Input parameters
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
NumPeaks = 5; % Number of resonance guesses

% Baron formatted vector of solution parameters
% Last num_res_guess values are binary switches for whether or not resonance exists
sol_w = zeros(1, 3 * NumPeaks);
for j=1:num_res_actual
    sol_w(3*(j-1)+1) = Gc(j);
    sol_w(3*(j-1)+2) = gn_square(j);
    sol_w(3*(j-1)+3) = Elevels(j);
end
for j=num_res_actual+1:NumPeaks
    sol_w(3*(j-1)+1) = 0;
    sol_w(3*(j-1)+2) = 0;
    sol_w(3*(j-1)+3) = mean(Elevels);
end


%% Create total cross section function

% Define energy grid
WE = exp_pw.E';  % Passing WE to WE is artifact of Jordan-Noah edits

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

% Use syndat data instead of Jordan's noisy data
WC = exp_pw.exp_trans';

if plotting
    % % Plot theoretical and experimental syndat data
    figure(1); clf
    errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
    plot(WE, trans_func(sol_w),'o', 'DisplayName', 'Matlab theo') ; hold on
    plot(WE, exp_pw.theo_trans, 'DisplayName', 'Syndat theo')
    legend()
end

%% Set Baron options

% chi2 = @(w) (trans_func(w)-WC) * inv(exp_cov) *  (trans_func(w)-WC)' ;
diag_cov = (diag(exp_cov))' ;
% chi2 = @(w) (trans_func(w)-WC) * inv_diag_cov *  (trans_func(w)-WC)' ;
fun_robust1=@(w) sum((trans_func(w)-WC).^2./diag_cov) ;
% fun_robust1=@(w) chi2(w) ;

% dof = length(WE)-1; 
% G = gamma(dof/2) ; 
% mypdf = @(w) ( 1/(2^(dof/2)*G) * chi2(w)^(dof/(2-1)) * exp(-chi2(w)/2) )  / chi2pdf(dof-2,dof);
% fun_robust1= @(w) -log10( mypdf(w) );

% insert min/max of Gc, gn_square, and energy 
Gc_bound = [min(Gc)*0.9 max(Gc)*1.1];
gn_square_bound = [min(gn_square)*0.9, max(gn_square)*1.1];
MinVec = [Gc_bound(1) gn_square_bound(1)  min(WE)];
MaxVec = [Gc_bound(2) gn_square_bound(2)  max(WE)];

lb=repmat(MinVec,1,NumPeaks);
ub=repmat(MaxVec,1,NumPeaks);

% baron runtime options
Options=baronset('threads',8,'PrLevel',0,'MaxTime',15*60, 'EpsA', fun_robust1(sol_w) ,'barscratch', sprintf('/home/nwalton1/reg_perf_tests/perf_tests/staticladder/baron_rev2/bar_%i/', isample));
xtype=squeeze(char(repmat(["C","C","C"],1,NumPeaks)))';

if initial_guess
    w0 = zeros(1, 3 * NumPeaks);
    for j=1:length(suggested_peaks)
        w0(3*(j-1)+1) = rand*(Gc_bound(2)-Gc_bound(1)) + Gc_bound(1);
        w0(3*(j-1)+2) = rand*(gn_square_bound(2)-gn_square_bound(1)) + gn_square_bound(1);
        w0(3*(j-1)+3) = suggested_peaks(j);
    end
else
    w0 = []; 
end

if plotting
    figure(1); clf
    errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
    plot(WE, trans_func(sol_w), 'DisplayName', 'Matlab theo') ; hold on
    % plot(WE, trans_func(w0), 'o', 'DisplayName', 'initial guess') 
    legend()
end

%% Baron solve

[w1,fval,ef,info]=baron(fun_robust1,[],[],[],lb,ub,[],[],[],xtype,w0, Options); % run baron

%% Plot results
if plotting
    figure(1); clf
    errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
    plot(WE, trans_func(sol_w), 'DisplayName', 'Matlab theo') ; hold on
    % plot(WE, exp_pw.theo_trans, 'DisplayName', 'Syndat theo'); hold on
    % plot(WE, trans_func(w0), 'DisplayName', 'initial guess') 
    plot(WE, trans_func(w1), 'DisplayName','Baron sol');
    legend()
    
    fprintf('SE solution: %f\n',fun_robust1(sol_w))
    fprintf('SE Baron: %f\n', fval)
end 

%% Write out results
tStop = toc(tStart) ; 
% h5writeatt(case_file,sprintf('/sample_%i', isample),'tfit',tStop)

% estimated parameter table
parameter_matrix = reshape(w1(1:NumPeaks*3),3,[])' ;
E = parameter_matrix(:,3) ; 
Gg = parameter_matrix(:,1)   *1e3 ; 
gnx2 = parameter_matrix(:,2) *1e3 ; 
tfit = [tStop; zeros(NumPeaks-1,1)];

parameter_estimate_table = table(E, Gg, gnx2, tfit);
writetable(parameter_estimate_table, sprintf('./par_est_%i.csv', isample))

end



function table_out = read_hdf5(case_file, dataset)

float_data = h5read(case_file, sprintf('%s/block0_values', dataset));
label_data = h5read(case_file, sprintf('%s/block0_items',dataset));

%remove whitespace
for i = 1:length(label_data)
    label_data(i) = deblank(label_data(i));
end

table_out = array2table(float_data','VariableNames',cellstr(label_data'));

end
