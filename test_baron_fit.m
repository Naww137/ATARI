%% loop over number of resonances in 

loop_peaks = [1 2 3 4 5 6 7 8];
% loop_peaks = [3 4 5 6 7 8];
loop_energies=[100 200 300 400 500 600 700 800 900 1200 1500];

total_time_matrix = zeros(length(loop_peaks), length(loop_energies));
final_SE_matrix = zeros(length(loop_peaks), length(loop_energies));


%%
for ipeak = 1:length(loop_peaks)
for ienergy = 1:length(loop_energies)


% user input
NumPeaks = loop_peaks(ipeak); % for baron solver
TrueNumPeaks = 1; % for true xs calculation

run_baron_bool = true ;
iterate_baron = true;

plotting = false ;
print_results_to_csv = true;

use_sammy_data = false ;
sample_new_resonance_parameters = false ;



if sample_new_resonance_parameters
    number_of_cases = 1 ;
    levels_per_case = TrueNumPeaks ; 
end

Energies = linspace(10, 1000, loop_energies(ienergy));


initial_vec = [];
% initial_vec = w;
% Options = baronset('threads',8,'PrLevel',1,'CutOff',5,'DeltaTerm',1,'EpsA',0.1,'MaxTime',2*60);
Options = baronset('threads',8,'PrLevel',0,'EpsA',0.01,'MaxTime',20*60);





%% nuclear properties
A = 62.929599; %Cu-63, number from cu63 input txt file
Constant = 0.002197; %sqrt(2Mn)/hbar; units of (10^(-12) cm sqrt(eV)^-1)
Ac = 0.67; % scattering radius 6.7 fermi expressed as 10^-12 cm
I = 1.5; % target angular Momentum
ii = 0.5; % incident angular momentum
l = 0;   % l=0 or s-wave spin group

s = I-ii; %
J = l+s; 
g = (2*J+1)/( (2*ii+1)*(2*I+1) );   % spin statistical factor g sub(j alpha)
pig = pi*g;

%l=0   or s-wave spin group energy dependent functions of the wave number
k = @(E) Constant*(A/(A+1))*sqrt(E);   % wave number
rho = @(E) k(E)*Ac;       % related to the center of mass momentum
P = @(E) rho(E);          % penatrability factor
S = 0; % for l=0
% average values for this single spin group
S0 = 2.3e-4; % strength function goes in to jordans derivation for reduced neutron width amplitude
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

%     sol_w = [417.501971239733	0.509754943249714	0.946324000441694	499.704338697496	0.502030277600131	5.48791199950379	815.972252536289	0.509617216760288	7.43229242626594];
    sol_w = [373.446529425425	0.458411490362457	3.55575669868274];
    true_xs_function = xs_SLBW_EGgGn(TrueNumPeaks,Energies);
    true_xs = true_xs_function(sol_w); 
    
end


if plotting
    figure(1); clf
    plot(Energies, true_xs, '.', 'DisplayName','True'); hold on
end


if run_baron_bool

    tic 

    WC = true_xs; 
    WE = Energies;
    xs_function = xs_SLBW_EGgGn(NumPeaks,Energies) ;
    [w, SE, fval] = run_baron(xs_function, NumPeaks, WC, WE, run_baron_bool, initial_vec, Options); 
    
    if iterate_baron
        while fval > 1
            Options = baronset('threads',8,'PrLevel',0,'EpsA',0.01,'MaxTime',10*60);
            [w, SE, fval] = run_baron(xs_function, NumPeaks, WC, WE, run_baron_bool, w, Options);

            if tic > 2*60
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
final_SE_matrix(ipeak,ienergy) = fval;

% end loops
end
end

if print_results_to_csv
    writematrix(total_time_matrix, 'total_time_matrix_1L.csv')
    writematrix(final_SE_matrix, 'final_SE_matrix_1L.csv')
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
