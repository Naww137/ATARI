%% user inputs 

% SOLVER OPTIONS
run_solver = true ;
iterate_solver = true;
normalize_range = false;
constraints = true;
solver = 'pswarm';

% OUTPUT OPTIONS
plotting = true ;
print_results_to_csv = true ;
running_on_cluster = true ;

% EXP DATA OPTIONS
TrueNumPeaks = %%%TrueNumPeaks%%% ; % for true xs calculation
use_sammy_data = false ;
sample_new_resonance_parameters = true ;
add_noise = false ; a = 1; b = 50;

% RUNTIME OPTIONS
maximum_total_time = 10*60; % 2*60*60; %
absolute_tolerance = 0.01; % absolute tolerance should be lower if transforming to [0,1]
print_out = 0;

initial_vec = [];
% initial_vec = w;



if running_on_cluster
    if strcmp(solver,'pswarm')
        addpath('/home/nwalton1/PSwarm');
        addpath('%%%main_directory%%%')
    end
else
    if strcmp(solver,'pswarm')
        addpath('/Users/noahwalton/Documents/GitHub/ATARI')
    end
end

if strcmp(solver,'pswarm')
    Options = PSwarm('defaults') ;
    Options.MaxObj = 1e7;
    Options.MaxIter = Options.MaxObj ;
    Options.CPTolerance = 1e-7;
    Options.DegTolerance = 1e-5;
    Options.IPrint = -1;
    Options.SearchType = 1 ;
    
    options_first_run = Options;
    options_iterations = Options;
elseif strcmp(solver,'baron')
%     Options = baronset('threads',8,'PrLevel',1,'CutOff',5,'DeltaTerm',1,'EpsA',0.1,'MaxTime',2*60);
    options_first_run = baronset('threads',8,'PrLevel',print_out,'EpsA',absolute_tolerance,'MaxTime',5*60);
    options_iterations = baronset('threads',8,'PrLevel',print_out,'EpsA',absolute_tolerance,'MaxTime',10*60);
end

partime = tic ;

%% start loop

w_vecs = [];

NumPeaks = %%%NumPeaks%%% ; % for baron solver
if sample_new_resonance_parameters
    number_of_cases = 1 ;
    levels_per_case = TrueNumPeaks ; 
end

starting_energy = 100 ;
% Energies = linspace(10, 1000, loop_energies(ienergy));


% nuclear properties
A = 62.929599; 
Constant = 0.002197; 
Ac = 0.67; 
I = 1.5; 
ii = 0.5; 
l = 0; 

s = I-ii; 
J = l+s; 
g = (2*J+1)/( (2*ii+1)*(2*I+1) ); 
pig = pi*g;

%l=0   or s-wave spin group energy dependent functions of the wave number
k = @(E) Constant*(A/(A+1))*sqrt(E); 
rho = @(E) k(E)*Ac;     
P = @(E) rho(E);
S = 0; % for l=0
% average values for this single spin group
S0 = 2.3e-4;
D0 = 100; %722; %s-wave; 722+-47 eV --- average level spacing <D>
avg_gn2 = ((4*D0.*S0)/(2*J+1))*((1/Constant)*((A+1)/A))/Ac; % Jordan derived this equation
avg_Gg = 0.5; %500 eV average capture width (known for an isotope)



%% get true_synthetic_cross section
if use_sammy_data

    case_basename = 'slbw_3L_allexp';%"slbw_testing_1L_noexp";

else

    if sample_new_resonance_parameters
        parameters_per_level = 3 ;
        parameters_per_case = TrueNumPeaks * parameters_per_level;
        for icase = 1:number_of_cases
            E_levels = sample_resonance_levels(starting_energy, TrueNumPeaks, D0);
            [Gg, gn] = sample_widths(TrueNumPeaks, avg_Gg, avg_gn2, P) ;
            Gn = 2.*P(E_levels).*gn.^2;
        end
        for ilevel = 1:TrueNumPeaks
            stride = parameters_per_level*(ilevel-1);
            sol_w(icase,1+stride:3+stride) = [E_levels(ilevel), Gg(ilevel), Gn(ilevel)];
        end 
    end

    Energies = linspace(starting_energy, max(E_levels)+D0/2, %%%EnergyPoints%%%);

    true_xs_function = xs_SLBW_EGgGn(TrueNumPeaks,Energies);
    true_xs = true_xs_function(sol_w); 

    if add_noise
        [std,exp_dat] = Noise(true_xs,a,b) ;
        absolute_tolerance = sum((true_xs - exp_dat).^2);

        if strcmp(solver,'baron')
            options_first_run.epsa = absolute_tolerance ;
            options_iterations.epsa = absolute_tolerance ;
        end

    else
        exp_dat = true_xs ;
    end

end


if plotting
    if running_on_cluster
        f = figure('visible','off');
    else
        figure(1); clf
    end
    plot(Energies, exp_dat, '.', 'DisplayName','True'); hold on
end



%% make a window to solve


WC = exp_dat; minWC=min(WC); maxWC=max(WC);
WE = Energies; minWE=min(WE); maxWE=max(WE);

if normalize_range
    WC_norm = (WC-minWC)./(maxWC-minWC);
    WE_norm = (WE-minWE)./(maxWE-minWE);
    if plotting
        figure(2); clf
        plot(WE_norm,WC_norm, '.','DisplayName','True Normed'); hold on
    end
end

if normalize_range
    WC_reconstructed = (WC_norm.*(maxWC-minWC)) + minWC ;
    WE_reconstructed = (WE_norm.*(maxWE-minWE)) + minWE;
    if plotting
        figure(1)
        plot(WE_reconstructed,WC_reconstructed, 'o', 'DisplayName', 'True Reconstructed')
    end
end

%% solve problem

if run_solver
    tstart = tic ;
    
    if normalize_range
        WE_to_solver = WE; %WE_norm;
        WC_to_solver = WC_norm;
    else
        WE_to_solver = WE;
        WC_to_solver = WC;
    end

%   Need to update peak spacing to be more robust if normalizing E range !!!
    xs_function = xs_SLBW_EGgGn(NumPeaks,WE);
%     xs_function_norm = xs_SLBW_EGgGn_norm(NumPeaks,WE,normalize_range, minWC, maxWC, minWE, maxWE);
    if normalize_range
        xs_func_to_solver = xs_function_norm ;
    else
        xs_func_to_solver = xs_function ;
    end


    if strcmp(solver,'pswarm')
        [w, SE, barout2] = run_PSwarm(xs_func_to_solver, NumPeaks, WC_to_solver, WE_to_solver, run_solver, initial_vec, constraints, options_first_run);
    elseif strcmp(solver,'baron')
        [w, SE, barout2] = run_baron(xs_func_to_solver, NumPeaks, WC_to_solver, WE_to_solver, run_solver, initial_vec, constraints, options_first_run);
    end

    % cannot reconstruct SE before entering the following tolerance loop
    % because baron will just keep terminating and outputting the same solution
    iterations = 0;
    if iterate_solver
        while SE > absolute_tolerance

            if strcmp(solver,'pswarm')
                [w, SE, barout2] = run_PSwarm(xs_func_to_solver, NumPeaks, WC_to_solver, WE_to_solver, run_solver, w, constraints, options_iterations);
            elseif strcmp(solver,'baron')
                [w, SE, barout2] = run_baron(xs_func_to_solver, NumPeaks, WC_to_solver, WE_to_solver, run_solver, w, constraints, options_iterations);
            end

                % for PSwarm, if SE is the same for multiple iterations,
                % restart with no intial guess might help
            iterations = iterations+1;
            if toc(tstart) > maximum_total_time
                break
            end
           
        end
    end
    total_time = toc(tstart) ;

    if plotting
        plot(Energies, xs_function(w),'DisplayName','Pred')
    end

end

if plotting
    xlabel('Energy'); ylabel('\sigma')
    legend()
    if running_on_cluster
        saveas(f, '﻿%%%figure_filename%%%');
    end
end


if SE < absolute_tolerance
    convergence = 1; %'converged' ;
else
    convergence = 0 ; %'not converged';
end

fileID = fopen('%%%output_filename%%%', 'a') ;
fprintf(fileID,'time (s) : %1$.5E\nSE : %2$.5E\niterations : %3$i\nconvergence : %4$i', total_time, SE, iterations, convergence) ;

if strcmp(solver,'baron')
    fprintf(fileID,'%1$s\n%2$s', barout2.BARON_Status, barout2.Model_Status);
end


















%%

function levels = sample_resonance_levels(E0, N_levels, avg_level_spacing)

levels=zeros(1,N_levels);
% spacing between resonances sampled from inv(cdf)
level_spacing=avg_level_spacing*sqrt(-4/pi*log(rand(1,N_levels))); 
% offsets starting E using random number
E0=E0-0.5*avg_level_spacing+avg_level_spacing*rand(1); 
levels(1)=E0+level_spacing(1);

for j=2:N_levels
    levels(j)=levels(j-1)+level_spacing(j);
end

end


function [Gg, gn] = sample_widths(N_levels, avg_Gg, avg_gn2, P)
% large DOF for capture width because many channels open for capture reaction
DOF = 500 ;

Gg=(avg_Gg/(DOF)).*chi2rnd(DOF,1,N_levels); %P=1
gn=normrnd(0,1,1,N_levels); 
gn2=(avg_gn2./mean(gn.^2)).*gn.^2;
gn=sqrt(gn2);

end


% ========
% function to add noise to synthetic true cross section
% ========
function [std,New_CrossSection]=Noise(CrossSection,a,b)
%Noise model to modify the cross section values that we are given from
%SAMMY, and report the std on those new cross sections values. These
%"represent" the exp cross sections that would be seen.

Trans_sigma=a*CrossSection+b;
NoisyTrans_sigma=normrnd(Trans_sigma,sqrt(Trans_sigma));
New_CrossSection=(NoisyTrans_sigma-b)/a; %
std=(sqrt(Trans_sigma))/a;
end

function reconstructed_SE = reconstruct_SE(xs_function, w, WC)
    reconstructed_SE = sum((xs_function(w)-WC).^2);
end
