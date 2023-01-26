%% Clear Block

close all
clear
clc
set(0,'DefaultFigureWindowStyle','docked')
figure(1)
options = optimoptions('fmincon','Display','none'); % suppress optimization descriptive output
rng('shuffle')

%%

% Use Bayesian update with covariance contraints
% Spin group fitting with gJ*Gn numerator then unfold

%% Control Block

% turn on/off plotting
plotting = true;
show_ans = false;

% number of repeated trials
num_trial = 1e0;

global xs_scale %#ok<GVMIS>
xs_scale     = 1e+2; % scaling factor to control size of true cross section
Gn_avg       = 1e-2; % average neutron width
Gg_avg       = 1e-2; % average   gamma width
num_n_dof    = 1e0;  % neutron degrees of freedom
num_g_dof    = 1e2;  %   gamma degrees of freedom
width_prob   = 1e-5; % fraction of width distribution to be cut-off in optimizaiton limits
Gn_threshold = 1e-5; %

chi_width    = 1;    % width in number of average resonance widths
fit_width    = 4;    % width in number of average resonance widths

%%

xs_max = xs_scale/4;

Gn_min = 0;
Gg_min = Gg_avg/num_g_dof*chi2inv(  width_prob,num_g_dof);
Gn_max = Gn_avg/num_n_dof*chi2inv(1-width_prob,num_n_dof);
Gg_max = Gg_avg/num_g_dof*chi2inv(1-width_prob,num_g_dof);

res_par_avg.num_n_dof  = num_n_dof ;
res_par_avg.num_g_dof  = num_g_dof ;
res_par_avg.Gn         = [Gn_min Gn_avg Gn_max];
res_par_avg.Gg         = [Gg_min Gg_avg Gg_max];
res_par_avg.Gt         = res_par_avg.Gn + res_par_avg.Gg;

error      = zeros(num_trial,1);
chi2_final = zeros(num_trial,1);
time       = zeros(num_trial,1);

for i_trial = 1:num_trial
    tic

    % number of resonaces in the true data set
    res_par_avg.num_res = 5;

    % number of data points per resonance
    points_per_res = 30;

    % number of data points
    num_E = points_per_res* res_par_avg.num_res;

    %% Import block

    % Set up true resonance parameters
    res_par_true = get_res_par_true(res_par_avg);

    % Set up data 
    % E: vector of energy points
    % D: experimental data
    % V: variance of experimental data
    [E,D,V] = get_data(res_par_true,num_E); % create data from true resonance parameters

    % Used later
    % T: theory computed from resonance parameters

    chi_width = chi_width*res_par_avg.Gt(2);
    chi_width = max(1,sum(E < chi_width));
    fit_width = fit_width*res_par_avg.Gt(2);
    fit_width = max(1,sum(E < fit_width));

    %%
    
    % initialize resonance parameters fit as 0 or an initial guess
    res_par_fit = initialize_res_par([],res_par_avg);

    while true

        E_index = find_min(res_par_fit,chi_width,E,D,V);               
        res_par_loc = initialize_res_par(E(E_index),res_par_avg);

        a = E_index-fit_width;
        b = E_index+fit_width;
        a = max(1,a);
        b = min(b,num_E);

        res_par_loc.Er(1) = E(a);
        res_par_loc.Er(3) = E(b);        

        i_res = 1;
        [x,lb,ub] = res_par2x(res_par_loc);
        obj = @(x)        get_chi(x2res_par(x,lb,ub),E,D-get_T(E,res_par_fit),V,i_res);        
        [x,~,exitflag] = fmincon(obj,x,[],[],[],[],lb,ub,[],options);
        res_par_loc = x2res_par(x,lb,ub);
        if res_par_loc.Gn(1,2) < Gn_threshold            
            break
        end
        
        res_par_fit = merge_res_par(res_par_loc,res_par_fit);
        [x,lb,ub] = res_par2x(res_par_fit);
        obj = @(x)        sum(get_chi(x2res_par(x,lb,ub),E,D,V));        
        [x,~,exitflag] = fmincon(obj,x,[],[],[],[],lb,ub,[],options);
        res_par_fit = x2res_par(x,lb,ub);
        make_plot(res_par_fit,res_par_true,E,D,V,plotting,show_ans,i_res)        
    end
end

%%

show_ans = true;
make_plot(res_par_fit,res_par_true,E,D,V,plotting,show_ans)        

disp(['Chi2 True = ',num2str(1/num_E*sum(((D-get_T(E,res_par_true)).^2./V)))])
disp(['Chi2 Fit  = ',num2str(1/num_E*sum(((D-get_T(E,res_par_fit )).^2./V)))])

%%



%%

% get true resonace parameters
function res_par = get_res_par_true(res_par_avg)
num_res         = res_par_avg.num_res  ;
Gn_avg          = res_par_avg.Gn(2)    ;
Gg_avg          = res_par_avg.Gg(2)    ;
num_n_dof       = res_par_avg.num_n_dof;
num_g_dof       = res_par_avg.num_g_dof;

res_par.Er = zeros(num_res,3);
res_par.Gn = zeros(num_res,3);
res_par.Gg = zeros(num_res,3);

res_par.Er(:,2) = rand(num_res,1);
res_par.Gn(:,2) = Gn_avg*chi2rnd(num_n_dof,[num_res,1])/num_n_dof;
res_par.Gg(:,2) = Gg_avg*chi2rnd(num_g_dof,[num_res,1])/num_g_dof;

% set to allow the calculation of chi2
res_par.Er(:,1) = 0;
res_par.Er(:,3) = 1;
end

% calculate cross section based on resonance parameters
function T = get_T(E,res_par)
global xs_scale %#ok<GVMIS>
T = zeros(size(E));
num_res = size(res_par.Er,1);
for i_res = 1:num_res
    Er = res_par.Er(i_res,2);
    Gn = res_par.Gn(i_res,2);
    Gg = res_par.Gg(i_res,2);
    T = T + xs_scale.*Gn*Gg./((E-Er).^2 + (Gn+Gg)^2);
end
end

% create data from true resonance parameters
function [E,D,V] = get_data(res_par_true,num_E)

% uniform points in energy between 0 and 1
E = linspace(0,1,num_E);

% true cross section
T = get_T(E,res_par_true);

% variance on the experimental data
V = T;

% noisy experimental data
D  = get_D(T,V);
end

% add noise to data
function D = get_D(T,V)
D = T + sqrt(V).*randn(size(T));
end

function res_par = initialize_res_par(Er,res_par_avg)
if isempty(Er)
    res_par.Er       = zeros(0,3);
    res_par.Gn       = zeros(0,3);
    res_par.Gg       = zeros(0,3);
else
    res_par.Er    = [Er Er Er];
    res_par.Gn    = res_par_avg.Gn;
    res_par.Gn(2) = eps;            % why floating point relative accuracy?

    res_par.Gg    = res_par_avg.Gg;
    nu = res_par_avg.num_g_dof;
    res_par.Gg(2) = res_par.Gg(2)*(nu-2)/nu;
end
end

function res_par = x2res_par(x,lb,ub)
lb = reshape(lb,[],3);
res_par.Er(:,1) = lb(:,1);
res_par.Gn(:,1) = lb(:,2);
res_par.Gg(:,1) = lb(:,3);

x = reshape(x,[],3);
res_par.Er(:,2) = x(:,1);
res_par.Gn(:,2) = x(:,2);
res_par.Gg(:,2) = x(:,3);

ub = reshape(ub,[],3);
res_par.Er(:,3) = ub(:,1);
res_par.Gn(:,3) = ub(:,2);
res_par.Gg(:,3) = ub(:,3);
end

function [x,lb,ub] = res_par2x(res_par)
lb = [
    res_par.Er(:,1)
    res_par.Gn(:,1)
    res_par.Gg(:,1)
    ];
x  = [
    res_par.Er(:,2)
    res_par.Gn(:,2)
    res_par.Gg(:,2)
    ];
ub = [
    res_par.Er(:,3)
    res_par.Gn(:,3)
    res_par.Gg(:,3)
    ];
end

function res_par = merge_res_par(res_par_loc,res_par_fit)
res_par.Er = [
    res_par_loc.Er
    res_par_fit.Er
    ];
res_par.Gn = [
    res_par_loc.Gn
    res_par_fit.Gn
    ];
res_par.Gg = [
    res_par_loc.Gg
    res_par_fit.Gg
    ];
end

function res_par = remove_resonance(res_par,i_res)
res_par.Er(i_res,:) = [];
res_par.Gn(i_res,:) = [];
res_par.Gg(i_res,:) = [];
end

function chi2_value = get_chi(res_par,E,D,V,i_res,fit)
T = get_T(E,res_par);
if nargin == 4
    num_res   = size(res_par.Er,1);
    chi2_value = zeros(num_res,1);
    for i_res = 1:num_res
        fit = res_par.Er(i_res,1) <= E & E <= res_par.Er(i_res,3);
        chi2_value(i_res) = sum((D(fit)-T(fit)).^2./V(fit));
        nu = sum(fit);
        chi2_value(i_res) = chi2pdf(chi2_value(i_res),nu)/chi2pdf(nu-2,nu);
    end
elseif nargin == 5
    fit = res_par.Er(i_res,1) <= E & E <= res_par.Er(i_res,3);
    chi2_value = sum((D(fit)-T(fit)).^2./V(fit));
    nu = sum(fit);
    chi2_value = chi2pdf(chi2_value,nu)/chi2pdf(nu-2,nu);
elseif nargin == 6
    fit = ~sum(fit,1);
    chi2_value = sum((D(fit)-T(fit)).^2./V(fit));
    nu = sum(fit);
    chi2_value = chi2pdf(chi2_value,nu)/chi2pdf(nu-2,nu);
else
    assert(false)
end
index = chi2_value < 0;
chi2_value(index) = eps;
chi2_value = -log10(chi2_value);
end

% check nonlinear contraint
function [const_inequality,const_equality] = nonlcon(res_par,E,D,V,best_chi)
chi2_value = get_chi(res_par,E,D,V);
const_inequality =  chi2_value - best_chi;
const_equality = 0; % optional equality contraint is always met
end

function make_plot(res_par_fit,res_par_true,E,D,V,plotting,show_ans,i_res)
if ~plotting
    return
end

clf

% fine energy grid to make pretty plots
E_fine = linspace(0,1,1e4);

% data
errorbar(E,D,sqrt(V),'k.')
xlim([0,1])
hold on
global xs_scale
plot([0,1],xs_scale/4*ones(1,2),'k--')

if nargin > 7
    % identify data points in boolean fit vector
    fit = res_par_fit.Er(i_res,1) <= E & E <= res_par_fit.Er(i_res,3);
    plot(E(fit),D(fit),'g.')
    plot(res_par_fit.Er(i_res,2)*ones(1,2),[0,xs_scale/4],'g-','LineWidth',3)
end

if show_ans
    num_res = size(res_par_true.Er,1);
    for i_res = 1:num_res
        Er = res_par_true.Er(i_res,2);
        Gn = res_par_true.Gn(i_res,2);
        Gg = res_par_true.Gg(i_res,2);
        Gt = Gn + Gg;
        T = get_T(Er,res_par_true);
        % true resonance energies
        plot(Er*ones(1,2),[0,T],'b','LineWidth',2)
        % true resonance width
        plot(Er+Gt*[-1,1],T/2*ones(1,2),'b','LineWidth',2)
    end
    % true cross section
    plot(E_fine,get_T(E_fine,res_par_true),'b','LineWidth',2)
end

num_res = size(res_par_fit.Er,1);
for i_res = 1:num_res
    rng(num_res-i_res+1)
    fit = res_par_fit.Er(i_res,1) <= E & E <= res_par_fit.Er(i_res,3);
    plot(E(fit),D(fit),'o')
    fill([...
        res_par_fit.Er(i_res,1) ...
        res_par_fit.Er(i_res,1) ...
        res_par_fit.Er(i_res,3) ...
        res_par_fit.Er(i_res,3) ...
        ], ...
        [ ...
        0 ...
        xs_scale/4 ...
        xs_scale/4 ...
        0 ...
        ],rand(1,3),'FaceAlpha',0.1)
    Er = res_par_fit.Er(i_res,2);
    Gn = res_par_fit.Gn(i_res,2);
    Gg = res_par_fit.Gg(i_res,2);
    Gt = Gn + Gg;
    T = get_T(Er,res_par_fit);
    % found resonance energies
    plot(Er*ones(1,2),[0,T],'r')
    % found resonance widths
    plot(Er+Gt*[-1,1],T/2*ones(1,2),'r')
end
rng('shuffle')
% found fit cross section
plot(E_fine,get_T(E_fine,res_par_fit),'r','LineWidth',2)

pause(1e-1)
end

function [E_index] = find_min(res_par,width,E,D,V)
num_E = length(E);

X = zeros(1,num_E);
for iE = 1:num_E
    min_index = max([1,iE-width]);
    max_index = min([num_E,iE+width]);
    if size(res_par.Er,1) > 0       
        res_par.Er(1,1) = E(min_index);        
        res_par.Er(1,3) = E(max_index);
        chi2_value = get_chi(res_par,E,D,V,1);
    else
        fit = true(1,num_E);
        fit(min_index:max_index) = false;
        chi2_value = get_chi(res_par,E,D,V,[],fit);
    end
    X(iE) = chi2_value;
end
[~,E_index] = max(X);
end
