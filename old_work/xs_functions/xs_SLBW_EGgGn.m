function xs = xs_SLBW_EGgGn(NumPeaks, WE)
    
    % Nuclear Parameters
    A=62.929599;
    Constant=0.002197; 
    Ac=0.67; 
    I=1.5; 
    ii=0.5; 
    l=0;  
    s=I-ii; 
    J=l+s;  %
    g=(2*J+1)/( (2*ii+1)*(2*I+1) );   
    pig=pi*g;
    
    %l=0   or s-wave spin group energy dependent functions of the wave number
    k=@(E) Constant*(A/(A+1))*sqrt(E);  
    rho=@(E) k(E)*Ac;    
    P=@(E) rho(E); % not using right now
    

    xs = @(w) 0;
    for jj=1:NumPeaks
        xs=@(w) xs(w)+( (w(3+3*(jj-1)).*w(2+3*(jj-1)))  ./ ( (WE-w(1+3*(jj-1))).^2  + ((w(3+3*(jj-1))+w(2+3*(jj-1)))./2).^2 ) );
    end
    xs = @(w) ((pig)./k(WE).^2).*xs(w) ;
% 

end