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


%% sample many realizations of resonance parameters for a single resonance case

% user input
number_of_cases = 100 ;
levels_per_case = 3 ; 

parameters_per_level = 3 ;
parameters_per_case = levels_per_case * parameters_per_level;

sammy_input_RM = zeros(number_of_cases, parameters_per_case);

starting_energy = 1700 ;

for icase = 1:number_of_cases

E_levels = sample_resonance_levels(starting_energy, levels_per_case, D0);
[Gg, gn] = sample_widths(levels_per_case, avg_Gg, avg_gn2, P) ;
Gn = 2.*P(E_levels).*gn.^2;

% stride through each resonance level
for ilevel = 1:levels_per_case
    stride = parameters_per_level*(ilevel-1);
    sammy_input_RM(icase,1+stride:3+stride) = [E_levels(ilevel), Gg(ilevel).*1e3, Gn(ilevel).*1e3]; % sammy takes widths in milli-e
end

end



%%
case_numbers = [1:number_of_cases]';
true_RM = [case_numbers, sammy_input_RM];

T = array2table(true_RM);
T.Properties.VariableNames(1) = "case";
for ilevel =1:levels_per_case
    stride = parameters_per_level*(ilevel-1);
    E="E%d"; Gg="Gg%d"; Gn="Gn%d";
    stride_title = [sprintf(E,ilevel), sprintf(Gg,ilevel), sprintf(Gn,ilevel)];
    T.Properties.VariableNames(2+stride:4+stride) = stride_title;
end

% stride = @(stride_location)  stride_location+parameters_per_level*(ilevel-1);
% Elevels_only = [T{:,"E1"},T{:,"E2"},T{:,"E3"},T{:,"E4"},T{:,"E5"}]; 
% Gg_only = [T{:,"Gg1"},T{:,"Gg2"},T{:,"Gg3"},T{:,"Gg4"},T{:,"Gg5"}];
% Gn_only = [T{:,"Gn1"},T{:,"Gn2"},T{:,"Gn3"},T{:,"Gn4"},T{:,"Gn5"}]; 

Elevels_only = [T{:,"E1"},T{:,"E2"},T{:,"E3"}]; 
Gg_only = [T{:,"Gg1"},T{:,"Gg2"},T{:,"Gg3"}];
Gn_only = [T{:,"Gn1"},T{:,"Gn2"},T{:,"Gn3"}];

disp('E Level Range min/max (eV)'); disp([min(Elevels_only,[],'all'), max(Elevels_only,[],'all')])
disp('Gg Range min/max (meV)'); disp([min(Gg_only,[],'all'), max(Gg_only,[],'all')])
disp('Gn Range min/max (meV)'); disp([min(Gn_only,[],'all'), max(Gn_only,[],'all')])

writetable(T,'true_parameters.csv')

%% put all of these into sammy inputs

% for icase = 1:number_of_cases
% 
%     E = sammy_input_RM(icase,1);
%     Gg = sammy_input_RM(icase,2);
%     Gn = sammy_input_RM(icase,3);
%     
%     fileID = fopen('template_1L.par','r');
%     string = fread(fileID,'*char')';
%     fclose(fileID);
% 
%     string = regexprep(string,'%%%E_level%%%',num2str(E,10));
%     string = regexprep(string, '%%%_Gg_%%%', num2str(Gg,9));
%     string = regexprep(string, '%%%_Gn_%%%', num2str(Gn,9));
%     
%     filename = "slbw_fitting_case" + num2str(icase) + ".par" ;
%     fid  = fopen(filename,'w');
%     fprintf(fid,'%s',string);
%     fclose(fid);
% 
% end



%% functions
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
