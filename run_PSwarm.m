function [w, fval, RunData] = run_PSwarm(xs_func, NumPeaks, WC, WE, run_solver_bool, initial_vec, constraints, Options)

fobj.ObjFunction = @(w) sum((xs_func(w)-WC).^2);
% SE = fun_robust;

% plot(WE,xs_func(true_w(1,:)))
w = [];
if run_solver_bool

    if constraints
        % constraints
        % min/max vec must align with the order of parameters
        % MinVec = [min(WE) min(true_w(:,2)) min(true_w(:,3))];
        % MaxVec = [max(WE) max(true_w(:,2)) max(true_w(:,3))];
        MinVec = [min(WE) 0 0];
        MaxVec = [max(WE) 1 100];
        RM_PerPeak = 3 ;
        TotalRM_PerWindow = NumPeaks*RM_PerPeak;
        TotalParm_PerWindow=NumPeaks*(RM_PerPeak);
        
        minimum = []; maximum = [];
        for jj=1:NumPeaks
            minimum = [minimum; MinVec'];
            maximum = [maximum; MaxVec'];
        end
        fobj.Variables = TotalParm_PerWindow ;
        fobj.LB = minimum;
        fobj.UB = maximum;
        
        InitialPopulation = [];
        if isempty(initial_vec)
        else
            InitialPopulation(1).x = initial_vec;
            % InitialPopulation(2).x = [2,3]; second guess
        end

        % could add other constraints
    
        [w, fval, RunData] = PSwarm(fobj,InitialPopulation,Options);

    else
        disp('run_PSwarm not set up for constraints = false')
        return
    end

end


end