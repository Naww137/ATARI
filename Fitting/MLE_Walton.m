% function MLE_Walton(case_file, isample)


addpath('/Users/noahwalton/software/PSwarmM_v2_1')
% Description
% baron_fit_rev2_pp for initial min(SE/diag(cov))
% Then solves likelihood ratios iteratively to get MLE solution
% solves for multiple significance levels with MLE to demonstrate
% hyperparameter tuning

tStart = tic ; 
initial_guess = true;
plotting = true ;

case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/perf_test_staticwindow_poleposition.hdf5';
isample = 2 ;

% Load data as a table
exp_pw = read_hdf5(case_file, sprintf('/sample_%i/exp_pw', isample)) ;
exp_cov = h5read(case_file, sprintf('/sample_%i/exp_cov', isample));
theo_par = read_hdf5(case_file, sprintf('/sample_%i/theo_par', isample)) ; 
theo_chi2 = (exp_pw.theo_trans-exp_pw.exp_trans)' * inv(exp_cov) *  (exp_pw.theo_trans-exp_pw.exp_trans) ;
theo_SE = sum((exp_pw.theo_trans-exp_pw.exp_trans).^2./diag(exp_cov));


if initial_guess
%     suggested_peaks = readmatrix(sprintf('./suggested_peaks_%i.csv', isample)); 
    suggested_peaks = read_hdf5(case_file, sprintf('/sample_%i/poles', isample)) ;
    suggested_peaks = suggested_peaks(suggested_peaks.peak_sq_divE > 3, :); 
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

%% Create total cross section function to solve

% Define energy grid
WE = exp_pw.E';  % Passing WE to WE is artifact of Jordan-Noah edits
WC = exp_pw.exp_trans';

% cutoff = 1 ;
% WE = WE(cutoff:end-cutoff); 
% exp_cov = exp_cov(cutoff:end-cutoff, cutoff:end-cutoff);
% WC = WC(cutoff:end-cutoff);

initial_peaks = suggested_peaks.E(and((suggested_peaks.E<max(WE)),(suggested_peaks.E>min(WE)))) ;
NumPeaks = length(initial_peaks); % Number of resonance guesses
trans_func = get_vectorized_trans_func(NumPeaks, WE, pig, A, Ac, Constant); 

%% get solution vector and function

% Extract resonance ladder values from Syndat
Elevels = theo_par.E';
Gc = 0.001 * theo_par.Gg';
gn_square = 0.001 * theo_par.gnx2';

% Number of resonances
num_res_actual = height(theo_par);
% NumPeaks = num_res_actual ; 
% trans_func = get_vectorized_trans_func(NumPeaks, WE, pig, A, Ac, Constant);

% Baron formatted vector of solution parameters
sol_w = zeros(1, 3 * num_res_actual);
for j=1:num_res_actual
    sol_w(3*(j-1)+1) = Gc(j);
    sol_w(3*(j-1)+2) = gn_square(j);
    sol_w(3*(j-1)+3) = Elevels(j);
end
trans_func_sol = get_vectorized_trans_func(num_res_actual, WE, pig, A, Ac, Constant);

%% plot
if plotting
    % % Plot theoretical and experimental syndat data
    figure(1); clf
    errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
    plot(WE, trans_func_sol(sol_w),'o', 'DisplayName', 'Matlab theo') ; hold on
    plot(exp_pw.E, exp_pw.theo_trans, 'DisplayName', 'Syndat theo')
    legend()
end

%% Setup initial min X2 solve

diag_cov = (diag(exp_cov))' ;
fun_robust_SE = @(w) sum((trans_func(w)-WC).^2./diag_cov) ;
% fun_robust_SE = @(w) sum((trans_func_sol(w)-WC).^2./diag_cov) ;

% [lb, ub, xtype] = get_constraints(length(sol_w)/3, WE, min(Gc)*0.9, 0.11, max(gn_square)*1.1) ;
[lb, ub, xtype] = get_constraints(NumPeaks, WE, min(Gc)*0.9, 0.11, max(gn_square)*1.1) ;
% baron runtime options
Options = baronset('threads',8,'PrLevel',1,'MaxTime',10*60) %, 'EpsA', fun_robust1(sol_w) ) %,'barscratch', sprintf('/home/nwalton1/reg_perf_tests/perf_tests/staticladder/baron_rev2/bar_%i/', isample));

if initial_guess
%     w0 = sol_w; 
    w0 = zeros(1, 3 * NumPeaks);
    for j=1:NumPeaks %length(suggested_peaks.E)
        w0(3*(j-1)+1) = 0.1; %rand*(Gc_bound(2)-Gc_bound(1)) + Gc_bound(1);
        w0(3*(j-1)+2) = 0.001; %rand*(gn_square_bound(2)-gn_square_bound(1)) + gn_square_bound(1);
        w0(3*(j-1)+3) = initial_peaks(j);
    end
else
    w0 = []; 
end

if plotting
    figure(1); clf
    errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
%     plot(WE, trans_func_sol(sol_w), 'DisplayName', 'Matlab theo') ; hold on
    plot(exp_pw.E, exp_pw.theo_trans, 'DisplayName', 'Syndat theo')
    plot(WE, trans_func(w0), 'DisplayName', 'initial guess') 
    legend()
end

%% Initial solve for min(X2)

[w1_SE,fval,ef,info]=baron(fun_robust_SE,[],[],[],lb,ub,[],[],[],xtype,w0, Options); % run baron

% [fobj, opt] = get_PSwarm_minchi_func(NumPeaks, WC, WE, trans_func, exp_cov, lb, ub);
% InitialPopulation = [];
% InitialPopulation(1).x = w0 ;
% % trans_func(w0)
% % fobj.ObjFunction(w0)
% [w1_SE, fval, RunData] = PSwarm(fobj,InitialPopulation,opt);
%%
calc_chi2(trans_func_sol(sol_w),WC,exp_cov)
chi2 = calc_chi2(trans_func(w1_SE),WC,diag(diag(exp_cov)));
chi2 = calc_chi2(trans_func(w1_SE),WC,exp_cov);
dof = length(WE)-length(w1_SE); 
SE_likelihood = chi2pdf(chi2, dof)
chi2pdf(dof-2, dof)

%% Plot results
if plotting
    figure(1); clf
    errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
%     plot(WE, trans_func_sol(sol_w), 'DisplayName', 'Matlab theo') ; hold on
    plot(exp_pw.E, exp_pw.theo_trans, 'DisplayName', 'Syndat theo'); hold on
    % plot(WE, trans_func(w0), 'DisplayName', 'initial guess') 
    plot(WE, trans_func(w1_SE), 'DisplayName','Baron sol');
    legend()
    
    fprintf('SE solution: %f\n',theo_SE)
    fprintf('SE Baron: %f\n', fval)
end 

%% Cuttoff edges of window after initial SE solve
cutoff = 5 ;
WE_inner = WE(cutoff:end-cutoff);
WC_inner = WC(cutoff:end-cutoff);
exp_cov_inner = exp_cov(cutoff:end-cutoff, cutoff:end-cutoff);

% un optimized likelihood
trans_func_new = get_vectorized_trans_func(length(w1_SE)/3, WE_inner, pig, A, Ac, Constant);
chi2 = calc_chi2(trans_func_new(w1_SE),WC_inner,exp_cov_inner);
dof = length(WE_inner)-length(w1_SE); 
SE_likelihood = chi2pdf(chi2, dof);

if SE_likelihood < 1e-10
    exit('Min(X2) solution has likelihood < 1e-10')
end

%% Solve chi2 for each number of resonances, sort by minimum decrease in L

full_parameter_matrix = reshape(w1_SE,3,[])' ;
full_parameter_matrix_sorted = zeros(size(full_parameter_matrix)); 
sort_index = zeros(1,height(full_parameter_matrix)); 

for ireduced = 1:height(full_parameter_matrix)

if ireduced == 1
    reduced_parameter_matrix = full_parameter_matrix ; 
else
    if length(index_max_L_temp) > 1
        full_parameter_matrix_sorted(ireduced-1:end, :) = reduced_parameter_matrix(index_max_L_temp, :);
        break
    else
%     [row,col] = find(full_parameter_matrix(:,:)==reduced_parameter_matrix(index_min_chi2_temp, :)) ;
    [row,col] = find(full_parameter_matrix(:,:)==reduced_parameter_matrix(index_max_L_temp, :)) ;
    sort_index(ireduced-1) = unique(row);
%     full_parameter_matrix_sorted(ireduced-1, :) = reduced_parameter_matrix(index_min_chi2_temp, :); 
%     reduced_parameter_matrix(index_min_chi2_temp, :) = [] ;
    full_parameter_matrix_sorted(ireduced-1, :) = reduced_parameter_matrix(index_max_L_temp, :); 
    reduced_parameter_matrix(index_max_L_temp, :) = [] ;
    end
end

% chi2_vals = zeros(1, height(reduced_parameter_matrix));
likelihood_vals = zeros(1, height(reduced_parameter_matrix));
for idrop = 1:height(reduced_parameter_matrix)
    reduced_parameter_matrix_minus1 = reduced_parameter_matrix;
    reduced_parameter_matrix_minus1(idrop, :) = [] ;
    parameter_vector_reduced = reshape(reduced_parameter_matrix_minus1', [], 1) ;
    trans_func_new = get_vectorized_trans_func(height(reduced_parameter_matrix_minus1), WE_inner, pig, A, Ac, Constant);
    chi2 = calc_chi2(trans_func_new(parameter_vector_reduced),WC_inner,exp_cov_inner);
    dof = length(WE_inner)-length(parameter_vector_reduced); 
    likelihood = chi2pdf(chi2, dof);
%     chi2_vals(1,idrop) = chi2 ;
    likelihood_vals(1,idrop) = likelihood ;
end
% index_min_chi2_temp = find(chi2_vals==min(chi2_vals)) ; 
max(likelihood_vals)
index_max_L_temp = find(likelihood_vals==max(likelihood_vals)) ; 
end

% [row,col] = find(full_parameter_matrix(:,:)==reduced_parameter_matrix(index_min_chi2_temp, :)) ;
% sort_index(ireduced) = unique(row);
% full_parameter_matrix_sorted(ireduced, :) = reduced_parameter_matrix(index_min_chi2_temp, :); 
% reduced_parameter_matrix(index_min_chi2_temp, :) = [] ;



%% Heuristically find minimum number of resonances for which I think I can get a ML solution

for ireduced = 1:height(full_parameter_matrix_sorted)

reduced_parameter_matrix = full_parameter_matrix_sorted(ireduced:end,:);
parameter_vector_reduced = reshape(reduced_parameter_matrix', [], 1) ;

trans_func_new = get_vectorized_trans_func(height(reduced_parameter_matrix), WE_inner, pig, A, Ac, Constant);
dof = length(WE_inner)-length(parameter_vector_reduced); 
chi2 = calc_chi2(trans_func_new(parameter_vector_reduced), WC_inner, exp_cov_inner) ;
un_optimized_likelihood = chi2pdf(chi2, dof);
if un_optimized_likelihood < 1e-10
    min_index = ireduced ;
    break
end

end

%%
reduced_parameter_matrix = full_parameter_matrix_sorted(min_index-4:end,:);
parameter_vector_reduced = reshape(reduced_parameter_matrix', [], 1) ;

trans_func_new = get_vectorized_trans_func(height(reduced_parameter_matrix), WE_inner, pig, A, Ac, Constant);
dof = length(WE_inner)-length(parameter_vector_reduced); 
chi2 = calc_chi2(trans_func_new(parameter_vector_reduced), WC_inner, exp_cov_inner) ;
un_optimized_likelihood = chi2pdf(chi2, dof)



%% Now starting from min index, perform LRTs iteratively adding parameters 
% until upper limit defined by minimum number of parameters necessary to
% acheive an ML solution

% calc_chi2(trans_func_new(w1_MLE), WC_inner, exp_cov_inner)
trans_func_new = get_vectorized_trans_func(height(full_parameter_matrix_sorted), WE_inner, pig, A, Ac, Constant);
calc_chi2(trans_func_new(w1_SE),WC_inner,exp_cov_inner)
% plot_a_solution(1, WE_inner, WC_inner, exp_cov_inner, trans_func_new, w1_MLE)
% plot_a_solution(1, WE_inner, WC_inner, exp_cov_inner, trans_func, w1_SE)

%%
[fun_robust_MLE, likelihood_func_new] = get_MLE_function(trans_func_new, WE_inner, WC_inner, exp_cov_inner, dof);
[lb, ub, xtype] = get_constraints(length(reduced_parameter_matrix), WE_inner, min(Gc)*0.9, 0.11, max(gn_square)*1.1) ;
% Options = baronset('threads',8,'PrLevel',1,'MaxTime',3*60) %, 'EpsA', fun_robust1(sol_w) ) %,'barscratch', sprintf('/home/nwalton1/reg_perf_tests/perf_tests/staticladder/baron_rev2/bar_%i/', isample));
% [w1_MLE,fval,ef,info]=baron(fun_robust_MLE,[],[],[],lb,ub,[],[],[],xtype,w1_SE, Options);

options = optimoptions(@fmincon,'MaxFunctionEvaluations',1e5);
[w1_MLE,fval] = fmincon(fun_robust_MLE, parameter_vector_reduced, [],[],[],[], lb, ub, [], options) ;
% chi2_func(parameter_vector_reduced)
% chi2_func(w1_MLE)

%%

[fobj, opt] = get_PSwarm_MLE_func(length(reduced_parameter_matrix), WC_inner, WE_inner, trans_func_new, exp_cov_inner, lb, ub, dof); 
InitialPopulation = [];
InitialPopulation(1).x = parameter_vector_reduced ;
% trans_func(w0)
% fobj.ObjFunction(w0)
global abs_termination; abs_termination = -log10(chi2pdf(dof-2,dof)*0.9) ; % must give absolute termination! or remove it in source code
[w1_MLE, fval, RunData] = PSwarm(fobj,InitialPopulation,opt);

%%
if plotting
    figure(1); clf
%     errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
    errorbar(WE_inner, WC_inner, sqrt(diag(exp_cov_inner)), '.', 'DisplayName', 'Syndat exp'); hold on
%     plot(WE, trans_func_sol(sol_w), 'DisplayName', 'Matlab theo') ; hold on
    plot(WE, exp_pw.theo_trans, 'DisplayName', 'Syndat theo'); hold on
    % plot(WE, trans_func(w0), 'DisplayName', 'initial guess') 
    plot(WE, trans_func(w1_SE), 'DisplayName','X2 sol');
    plot(WE_inner, trans_func_new(w1_MLE), 'DisplayName','ML sol');
    legend()
    
    fprintf('MLE of SE solution: %f\n',fun_robust_MLE(w1_SE))
    fprintf('MLE Solved: %f\n', fun_robust_MLE(w1_MLE))

    figure(2); clf
    x = linspace(0,1000,1000);
    plot(x, chi2pdf(x,dof))
    xline(chi2_func(w1_MLE))
end 

%% Now perform likelihood ratio tests to eliminate resonances

% for p = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25]
p = 0.05 ; 

% sortrows(suggested_peaks, 'peak_sq_divE')
drop_resonances = true ;
dropped_resonances = 1 ;

parameter_matrix = reshape(w1_MLE,3,[])' ;
parameter_matrix_sorted_by_gnx2 = sortrows(parameter_matrix, 2) ;

% null = parameter_matrix_sorted_by_gnx2 ;
% alt = parameter_matrix_sorted_by_gnx2 ;

likelihood_alt = likelihood_func(w1_MLE); 

figure(1); clf
errorbar(WE_inner, WC_inner, sqrt(diag(exp_cov_inner)), '.', 'DisplayName', 'Syndat exp'); hold on
plot(WE, exp_pw.theo_trans); legend();

while drop_resonances
%     [D, LD, null, alt] = calculate_deviance(null, alt, dropped_resonances, length(WE), NumPeaks, chi2_func) ; 
    
    alt = parameter_matrix_sorted_by_gnx2(dropped_resonances:end, :); 
    parameter_vector_alt = reshape(alt', [], 1) ;
    npar_alt = length(parameter_vector_alt);
    
    null = parameter_matrix_sorted_by_gnx2(1+dropped_resonances:end,:);
    parameter_vector_null = reshape(null', [], 1) ;
    npar_null = length(parameter_vector_null) ; 
    
    ndat = length(WE_inner) ; 
    dof_null = ndat - npar_null ;
    dof_alternative = ndat - npar_alt ;
    
    % find MLE for new null 
    trans_func_new = get_vectorized_trans_func(height(null), WE_inner, pig, A, Ac, Constant);
    [fun_robust_MLE, likelihood_func_new] = get_MLE_function(trans_func_new, WE_inner, WC_inner, exp_cov_inner, dof_null);
    [lb, ub, xtype] = get_constraints(height(null), WE_inner, min(Gc)*0.9, max(Gc)*1.1, max(gn_square)*1.1) ;
    % Options = baronset('threads',8,'PrLevel',1,'MaxTime',3*60) %, 'EpsA', fun_robust1(sol_w) ) %,'barscratch', sprintf('/home/nwalton1/reg_perf_tests/perf_tests/staticladder/baron_rev2/bar_%i/', isample));
    % [w1_MLE,fval,ef,info]=baron(fun_robust_MLE,[],[],[],lb,ub,[],[],[],xtype,w1_SE, Options);
    [parameter_vector_null_MLE,fval] = fmincon(fun_robust_MLE, parameter_vector_null, [],[],[],[], lb, ub) ;
    
    likelihood_null = likelihood_func_new(parameter_vector_null_MLE); 
    
    D = 2*( log(likelihood_alt) - log(likelihood_null) )
    LD = 1 - chi2cdf(D,npar_alt-npar_null)
    
    if LD < p
        drop_resonances = false;
    end
   
    dropped_resonances = dropped_resonances + 1; 
    
    likelihood_alt = likelihood_null ;

%     plot(WE, trans_func(parameter_vector_alt), 'DisplayName', 'Alternative') 
    plot(WE_inner, trans_func_new(parameter_vector_null), 'DisplayName','Null');

end

parameter_vector_null = reshape(null', [], 1) ;
parameter_vector_alt = reshape(alt', [], 1) ;

%% Plot results
if plotting
    figure(1); clf
%     errorbar(WE, WC, sqrt(diag(exp_cov)), '.', 'DisplayName', 'Syndat exp'); hold on
    errorbar(WE_inner, WC_inner, sqrt(diag(exp_cov_inner)), '.', 'DisplayName', 'Syndat exp'); hold on
    plot(WE, exp_pw.theo_trans)
%     plot(WE, trans_func(parameter_vector_alt), 'DisplayName', 'Alternative') 
    plot(WE_inner, trans_func_new(parameter_vector_null), 'DisplayName','Null');
    legend()
end 
%%
% Write out results
tStop = toc(tStart) ; 
% h5writeatt(case_file,sprintf('/sample_%i', isample),'tfit',tStop)

% estimated parameter table
parameter_matrix = reshape(parameter_vector_null(1:NumPeaks*3),3,[])' ;
E = parameter_matrix(:,3) ; 
Gg = parameter_matrix(:,1)   *1e3 ; 
gnx2 = parameter_matrix(:,2) *1e3 ; 
tfit = [tStop; zeros(NumPeaks-1,1)];

parameter_estimate_table = table(E, Gg, gnx2, tfit);
writetable(parameter_estimate_table, sprintf('./par_est_%i_%f.csv', isample, p))

% end 

% 'end' for entire script as function
% end


%% Functions

function table_out = read_hdf5(case_file, dataset)

float_data = h5read(case_file, sprintf('%s/block0_values', dataset));
label_data = h5read(case_file, sprintf('%s/block0_items',dataset));

%remove whitespace
for i = 1:length(label_data)
    label_data(i) = deblank(label_data(i));
end

table_out = array2table(float_data','VariableNames',cellstr(label_data'));

end

function chi2 = calc_chi2(a,b,cov)
    chi2 =(a-b) * inv(cov) * (a-b)' ;
end

function [fun_robust_MLE, likelihood_func, chi2_func] = get_MLE_function(trans_func_new, WE, WC, exp_cov, dof)
inv_exp_cov = inv(exp_cov) ; 
chi2_func = @(w) (trans_func_new(w)-WC) * inv_exp_cov *  (trans_func_new(w)-WC)' ;
G = gamma(dof/2) ;
mychipdf = @(w) ( 1/(2^(dof/2)*G)*chi2_func(w)^(dof/2-1)*exp(-chi2_func(w)/2) );
likelihood_func = @(w) mychipdf(w);
fun_robust_MLE = @(w) -log10( likelihood_func(w) ) ;
end

function [fobj, opt] = get_PSwarm_minchi_func(NumPeaks, WC, WE, trans_func, exp_cov, lb, ub)

fobj.ObjFunction = @(w) sum((trans_func(w)-WC).^2./diag(exp_cov)');
fobj.Variables = NumPeaks*3 ;
fobj.LB = lb;
fobj.UB = ub;

opt=PSwarm('defaults') ;
opt.IPrint = 100; 
opt.MaxObj = 1e7;
opt.MaxIter = 1e7 ; %opt.MaxObj ;
opt.CPTolerance = 1e-7;
opt.DegTolerance = 1e-8;
end

function [fobj, opt] = get_PSwarm_MLE_func(NumPeaks, WC, WE, trans_func, exp_cov, lb, ub, dof)

fobj.ObjFunction = @(w) -log10( chi2pdf( (trans_func(w)-WC)*inv(exp_cov)* (trans_func(w)-WC)', dof) ) ;
fobj.Variables = NumPeaks*3 ;
fobj.LB = lb;
fobj.UB = ub;

opt=PSwarm('defaults') ;
opt.IPrint = 100; 
opt.MaxObj = 5e4;
opt.MaxIter = 1e5 ; %opt.MaxObj ;
opt.CPTolerance = 1e-5; %1e-7;
opt.DegTolerance = 1e-5;
end



function trans_func = get_vectorized_trans_func(NumPeaks, WE, pig, A, Ac, Constant)

% Functions of energy
k=@(E) Constant*(A/(A+1))*sqrt(E);   % wave number
rho=@(E) k(E)*Ac;       % related to the center of mass momentum
P=@(E) rho(E);          % penatrability factor

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
end


function [lb, ub, xtype] = get_constraints(NumPeaks, WE, min_Gc, max_Gc, max_gn_square)

% insert min/max of Gc, gn_square, and energy 
Gc_bound = [min_Gc, max_Gc]; 
gn_square_bound = [0, max_gn_square];
MinVec = [Gc_bound(1) gn_square_bound(1)  min(WE)-5];
MaxVec = [Gc_bound(2) gn_square_bound(2)  max(WE)+5];

lb=repmat(MinVec,1,NumPeaks);
ub=repmat(MaxVec,1,NumPeaks);

xtype=squeeze(char(repmat(["C","C","C"],1,NumPeaks)))';
end


function [D, LD, null, alt] = calculate_deviance(null, alt, dropped_resonances, ndat, NumPeaks, chi2_func)

if dropped_resonances == 1

else
    alt(dropped_resonances-1, 2) = 0.0 ; 
end
null(dropped_resonances,2) = 0.0 ; 

parameter_vector_null = reshape(null', NumPeaks*3, 1) ;
parameter_vector_alternative = reshape(alt', NumPeaks*3, 1) ;

% chi2 = @(w) (trans_func(w)-WC) * inv(exp_cov) *  (trans_func(w)-WC)' ;

npar_null = length(parameter_vector_null) - (3*dropped_resonances) ;
npar_alternative = length(parameter_vector_null) - (3*dropped_resonances-1) ;
dof_null = ndat - npar_null ;
dof_alternative = ndat - npar_alternative ;

D = 2*( log(chi2pdf(chi2_func(parameter_vector_alternative),dof_alternative)) - log(chi2pdf(chi2_func(parameter_vector_null),dof_null)) );
LD = chi2pdf(D,npar_alternative-npar_null) ; 
end


function [] = plot_a_solution(fignum, WE, WC, cov, trans_func, solution)

figure(fignum);
errorbar(WE, WC, sqrt(diag(cov)), '.', 'DisplayName', 'Syndat exp'); hold on
plot(WE, trans_func(solution), 'DisplayName','sol');
legend()
end