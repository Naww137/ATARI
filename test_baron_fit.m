%% loop over number of resonances in 

% loop_peaks = [1 2 3 4 5 6 7 8];
loop_peaks = [3 4 5 6 7 8];
% loop_peaks = [1];

% loop_energies=[500];
loop_energies=[100 200 300 400 500 600 700 800 900 1200 1500];


%% user inputs 

% SOLVER OPTIONS
run_baron_bool = true ;
iterate_baron = true;
normalize_range = true;

% OUTPUT OPTIONS
plotting = false ;
print_results_to_csv = true ;

% EXP DATA OPTIONS
TrueNumPeaks = 1; % for true xs calculation
use_sammy_data = false ;
sample_new_resonance_parameters = false ;

% BARON RUNTIME OPTIONS
maximum_total_time = 2*60*60; % 5*60; %
absolute_tolerance = 0.0001; % absolute tolerance should be lower if transforming to [0,1]
print_out = 0;

initial_vec = [];
% initial_vec = w;
% Options = baronset('threads',8,'PrLevel',1,'CutOff',5,'DeltaTerm',1,'EpsA',0.1,'MaxTime',2*60);
options_first_run = baronset('threads',8,'PrLevel',print_out,'EpsA',absolute_tolerance,'MaxTime',20*60);
options_iterations = baronset('threads',8,'PrLevel',print_out,'EpsA',absolute_tolerance,'MaxTime',10*60);





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
Energies = linspace(10, 1000, loop_energies(ienergy));


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
D0 = 300; %722; %s-wave; 722+-47 eV --- average level spacing <D>
avg_gn2 = ((4*D0.*S0)/(2*J+1))*((1/Constant)*((A+1)/A))/Ac; % Jordan derived this equation
avg_Gg = 0.5; %500 eV average capture width (known for an isotope)



%% get true_synthetic_cross section
if use_sammy_data

    case_basename = 'slbw_3L_allexp';%"slbw_testing_1L_noexp";
    interface_directory = "/Users/noahwalton/Library/Mobile Documents/com~apple~CloudDocs/Research Projects/Resonance Fitting/sammy/";
    case_syndat_dir = strcat(interface_directory,case_basename,"/synthetic_data/");
    
    true_parms = readtable(strcat(interface_directory,case_basename,"/true_parameters.csv"));
    true_w = [true_parms{1,2:1+(NumPeaks*3)}].*repmat([1, 1e-3, 1e-3],1,NumPeaks); % widths are in meV!!!
%     sol_w = [true_w(2) true_w(3) true_w(1) true_w(5) true_w(6) true_w(4) true_w(8) true_w(9) true_w(7) true_w(11) true_w(12) true_w(10) true_w(14) true_w(15) true_w(13)];
    sol_w = true_w;

    filename = "syndat_"+string(1);
    filepath = strcat(case_syndat_dir,filename);
    synth_exp_dat = readmatrix(filepath, 'FileType','text');
    Energies = synth_exp_dat(415:2200,1)' ;
    true_xs = synth_exp_dat(415:2200,2)';
    % Energies = synth_exp_dat(:,1)' ;
    % true_xs = synth_exp_dat(:,2)';

else

    if sample_new_resonance_parameters
        parameters_per_level = 3 ;
        parameters_per_case = TrueNumPeaks * parameters_per_level;
        starting_energy = min(Energies) ;
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

%     for perfromances testing, the following two sets of parameters were used
    sol_w = [417.501971239733	0.509754943249714	0.946324000441694	499.704338697496	0.502030277600131	5.48791199950379	815.972252536289	0.509617216760288	7.43229242626594];
%     sol_w = [373.446529425425	0.458411490362457	3.55575669868274];

    true_xs_function = xs_SLBW_EGgGn(TrueNumPeaks,Energies);
    true_xs = true_xs_function(sol_w); 
    
end


if plotting
    figure(1); clf
    plot(Energies, true_xs, '.', 'DisplayName','True'); hold on
end



%% make a window to solve

WC = true_xs; minWC=min(WC); maxWC=max(WC);
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
    xs_function = xs_SLBW_EGgGn(NumPeaks,WE);
    xs_function_norm = xs_SLBW_EGgGn_norm(NumPeaks,WE,normalize_range, minWC, maxWC, minWE, maxWE);
    if normalize_range
        xs_func_to_baron = xs_function_norm ;
    else
        xs_func_to_baron = xs_function ;
    end

    [w, SE, barout2] = run_baron(xs_func_to_baron, NumPeaks, WC_to_baron, WE_to_baron, run_baron_bool, initial_vec, options_first_run);
    % cannot reconstruct SE before entering the following tolerance loop
    % because baron will just keep terminating and outputting the same solution
    if iterate_baron
        while SE > absolute_tolerance
            [w, SE, barout2] = run_baron(xs_func_to_baron, NumPeaks, WC_to_baron, WE_to_baron, run_baron_bool, w, options_iterations);
            if toc > maximum_total_time
                break
            end
        end
    end
    total_time = toc ;

    if plotting
        plot(Energies, xs_function(w),'DisplayName','Baron')
    end

end

if plotting
    xlabel('Energy'); ylabel('\sigma')
    legend()
end

total_time_matrix(ipeak,ienergy) = total_time;
final_SE_matrix(ipeak,ienergy) = SE;
baron_stat{ipeak,ienergy} = barout2.BARON_Status;
model_stat{ipeak,ienergy} = barout2.Model_Status;
if normalize_range
    final_SE_reconstructed(ipeak,ienergy) = reconstruct_SE(xs_function, w, WC);
end

% end loops
end
end

if print_results_to_csv
    writematrix(total_time_matrix, 'total_time_matrix_3L_norm.csv')
    writematrix(final_SE_matrix, 'final_SE_matrix_3L_norm.csv')
    writecell(baron_stat, 'baron_stat_matrix_3L_norm.csv')
    writecell(model_stat, 'model_stat_matrix_3L_norm.csv')
    if normalize_range
        writematrix(final_SE_reconstructed,'final_SE_reconstructed_3L.csv');
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
