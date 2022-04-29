%% loop over number of resonances in 

% loop_peaks = [1 2 3 4 5 6 7 8];
% loop_peaks = [3 4 5 6 7 8];
loop_peaks = [1];

loop_energies=[200];
% loop_energies=[100 200 300 400 500 600 700 800 900 1200 1500];


%% user inputs 

% SOLVER OPTIONS
run_baron_bool = true ;
iterate_baron = false;
normalize_range = false;
constraints = false;
give_solution = false;
fudge = 0.9;

% OUTPUT OPTIONS
plot_local = true ;
plot_cluster = false;
print_results_to_csv = false ;

% EXP DATA OPTIONS
TrueNumPeaks = 1; % for true xs calculation
use_sammy_data = false ;
sample_new_resonance_parameters = false ;
polynomial_terms = 4;

% BARON RUNTIME OPTIONS
maximum_total_time = 10*60; % 2*60*60; %
absolute_tolerance = 0.01; % absolute tolerance should be lower if transforming to [0,1]
print_out = 1;

initial_vec = [];
% initial_vec = w;
% Options = baronset('threads',8,'PrLevel',1,'CutOff',5,'DeltaTerm',1,'EpsA',0.1,'MaxTime',2*60);
options_first_run = baronset('threads',8,'PrLevel',print_out,'EpsA',absolute_tolerance,'MaxTime',.5*60);
options_iterations = baronset('threads',8,'PrLevel',print_out,'EpsA',absolute_tolerance,'MaxTime',5*60);

if polynomial_terms == 3
    polynomial_coefficients = [-782.996974515679	366.628693805771	-42.6132398273227];
elseif polynomial_terms == 4
    polynomial_coefficients = [-782.996974515679	366.628693805771	-42.6132398273227 0];
end


if plot_cluster && plot_local
    disp('WARNING: plot local & plot cluster boolian are TRUE')
    return
end
% if TrueNumPeaks ~= NumPeaks
%     disp('You are fitting with more peaks than there are, do you wish to continue?')


%% start loop
total_time_matrix = zeros(length(loop_peaks), length(loop_energies));
final_SE_matrix = zeros(length(loop_peaks), length(loop_energies));
final_SE_reconstructed =zeros(length(loop_peaks), length(loop_energies));
baron_stat = num2cell(zeros(length(loop_peaks), length(loop_energies)));
model_stat = num2cell(zeros(length(loop_peaks), length(loop_energies)));

for ipeak = 1:1%length(loop_peaks)
for ienergy = 1:1%length(loop_energies)


NumPeaks = loop_peaks(ipeak); % for baron solver
if sample_new_resonance_parameters
    number_of_cases = 1 ;
    levels_per_case = TrueNumPeaks ; 
end
% Energies = linspace(10, 1000, loop_energies(ienergy));
Energies = linspace(4.2, 4.8, loop_energies(ienergy));


% nuclear properties
% A = 62.929599; 
% Constant = 0.002197; 
% Ac = 0.67; 
% I = 1.5; 
% ii = 0.5; 
% l = 0; 
% 
% s = I-ii; 
% J = l+s; 
% g = (2*J+1)/( (2*ii+1)*(2*I+1) ); 
% pig = pi*g;
% 
% %l=0   or s-wave spin group energy dependent functions of the wave number
% k = @(E) Constant*(A/(A+1))*sqrt(E); 
% rho = @(E) k(E)*Ac;     
% P = @(E) rho(E);
% S = 0; % for l=0
% % average values for this single spin group
% S0 = 2.3e-4;
% D0 = 300; %722; %s-wave; 722+-47 eV --- average level spacing <D>
% avg_gn2 = ((4*D0.*S0)/(2*J+1))*((1/Constant)*((A+1)/A))/Ac; % Jordan derived this equation
% avg_Gg = 0.5; %500 eV average capture width (known for an isotope)



%% get true_synthetic_cross section
if use_sammy_data

    disp('WARNING: need to change use sammy data to use reconstructed U8 data')

else

    if sample_new_resonance_parameters

        disp('WARNING: Need to add nuclear parameters to properly sample U8 Resonances')
        return

        if polynomial_terms > 0
            disp('WARNING: Need to add code to sample polynomial terms')
%             return
        end

        parameters_per_level = 4 ;
        parameters_per_case = TrueNumPeaks * parameters_per_level;
        starting_energy = min(Energies) ;
        for icase = 1:number_of_cases
            E_levels = sample_resonance_levels(starting_energy, TrueNumPeaks, D0);
            [Gg, gn] = sample_widths(TrueNumPeaks, avg_Gg, avg_gn2, P) ;
            Gn = 2.*P(E_levels).*gn.^2;
        end

        widths = Gg + gn;

        p = []; r = []; %p = 10+0.1i; r = 2*exp(3*pi/2*1i);
        for ilevel = 1:TrueNumPeaks
            p = [p, Elevels(ilevel)+widths(ilevel)*1i]; 
            r = [r, 2*exp(3*pi/2*1i)];
        end
        ir = imag(r);
        rr = real(r);
        ip = imag(p);
        rp = real(p);
        sol_w = [];
        for ilevel = 1:NumPeaks
            sol_w = [sol_w rr(ilevel),ir(ilevel)*ip(ilevel),rp(ilevel),ip(ilevel)^2];
        end

    end

%     for perfromances testing, the following set of parameters is hard coded 
    if TrueNumPeaks == 1
        sol_w = [-0.0572641867005426	-1.81466913738840	4.56853649988364	3.25249651413280e-06];
    elseif TrueNumPeaks == 2
        sol_w = [-0.0572641867005426	-1.81466913738840	4.56853649988364	3.25249651413280e-06 -0.0572641867005426	-1.81466913738840	4.2 3.25249651413280e-06];
    end

    true_xs_function = xs_pole(TrueNumPeaks,Energies);
    if polynomial_terms > 0
        poly_index = 4*TrueNumPeaks; z = Energies;
        sol_w = [sol_w polynomial_coefficients];
        if polynomial_terms == 3
            true_xs_function = @(w) true_xs_function(w) + w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3 ;
        elseif polynomial_terms == 4
            true_xs_function = @(w) true_xs_function(w) + w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3 + w(poly_index+4).*z.^4 ;
        end
    end
    true_xs = true_xs_function(sol_w); 
    
end


if plot_local
    figure(1); clf
    semilogy(Energies, true_xs, 'DisplayName','True'); hold on
end
if plot_cluster
    myfig = figure('Visible', 'off'); hold on
%     set(gca, 'YScale', 'log')
%     set(gca, 'XScale', 'log')
    plot(Energies, true_xs,'.','DisplayName','true')
end

figure(1)
[std,true_xs]=Noise(true_xs,1,40);
plot(Energies, true_xs, '.', 'DisplayName','True')

%% make a window to solve

WC = true_xs; minWC=min(WC); maxWC=max(WC);
WE = Energies; minWE=min(WE); maxWE=max(WE);

if normalize_range
    WC_norm = (WC-minWC)./(maxWC-minWC);
    WE_norm = (WE-minWE)./(maxWE-minWE);
    if plot_local
        figure(2); clf
        plot(WE_norm,WC_norm, '.','DisplayName','True Normed'); hold on
    end
end

if normalize_range
    WC_reconstructed = (WC_norm.*(maxWC-minWC)) + minWC ;
    WE_reconstructed = (WE_norm.*(maxWE-minWE)) + minWE;
    if plot_local
        figure(1)
        plot(WE_reconstructed,WC_reconstructed, 'o', 'DisplayName', 'True Reconstructed')
    end
end

%% solve problem

if run_baron_bool
    tic 
    
    if normalize_range
        WE_to_baron = WE; %WE_norm;
        WC_to_baron = WC_norm;
    else
        WE_to_baron = WE;
        WC_to_baron = WC;
    end

%   Need to update peak spacing to be more robust if normalizing E range !!!
    xs_function = xs_pole(NumPeaks,WE);
    if polynomial_terms > 0
        poly_index = 4*NumPeaks;
        if polynomial_terms == 3
            xs_function = @(w) xs_function(w) + w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3 ;
        elseif polynomial_terms == 4
            xs_function = @(w) xs_function(w) + w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3+ w(poly_index+4).*z.^4 ;
        end
    end
%     xs_function_norm = xs_SLBW_EGgGn_norm(NumPeaks,WE,normalize_range, minWC, maxWC, minWE, maxWE);
    if normalize_range
        disp('WARNING: Need to add and xs_pol_norm function to use a normalized xs')
        xs_func_to_baron = @(w) (xs_function(w)-minWC)./(maxWC-minWC);
    else
        xs_func_to_baron = xs_function ;
    end

    if give_solution
        initial_vec = sol_w.*fudge;

    end


    [w, SE, barout2] = run_baron(xs_func_to_baron, NumPeaks, WC_to_baron, WE_to_baron, run_baron_bool, initial_vec, polynomial_terms, constraints, options_first_run);
    % cannot reconstruct SE before entering the following tolerance loop
    % because baron will just keep terminating and outputting the same solution
    if iterate_baron
        while SE > absolute_tolerance
            [w, SE, barout2] = run_baron(xs_func_to_baron, NumPeaks, WC_to_baron, WE_to_baron, run_baron_bool, w, polynomial_terms, constraints, options_iterations);
            if toc > maximum_total_time
                break
            end
        end
    end
    total_time = toc ;

    if plot_local || plot_cluster
        plot(Energies, xs_function(w),'DisplayName','Baron')
    end
%     if plot_cluster
%         plot(Energies, xs_function(w),'DisplayName','Baron')
%     end

    total_time_matrix(ipeak,ienergy) = total_time;
    final_SE_matrix(ipeak,ienergy) = SE;
    baron_stat{ipeak,ienergy} = barout2.BARON_Status;
    model_stat{ipeak,ienergy} = barout2.Model_Status;
    if normalize_range
        final_SE_reconstructed(ipeak,ienergy) = reconstruct_SE(xs_function, w, WC);
    end

end

if plot_local
    xlabel('Energy'); ylabel('\sigma')
    legend()
end
if plot_cluster
    xlabel('Energy'); ylabel('\sigma')
    legend()
    saveas(myfig,'figure.png');
end

% writing to csv within loop, slower, but will give results while running
if print_results_to_csv
    writematrix(total_time_matrix, 'total_time_matrix_2L.csv')
    writematrix(final_SE_matrix, 'final_SE_matrix_2L.csv')
    writecell(baron_stat, 'baron_stat_matrix_2L.csv')
    writecell(model_stat, 'model_stat_matrix_2L.csv')
    if normalize_range
        writematrix(final_SE_reconstructed,'final_SE_reconstructed_3L.csv');
    end
end


% end loops
end
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
