function xs = xs_SLBW(energies, E_levels, Gg, gn, pig, P, k)

xs = zeros(1,length(energies));
for jj = 1:length(E_levels)
    xs = xs + (pig./k(energies).^2).*((2*P(energies)*gn(jj)^2*Gg(jj)) ...
        ./((energies-E_levels(jj)).^2+((2*P(energies)*gn(jj)^2+Gg(jj))/2).^2)) ;
end

end