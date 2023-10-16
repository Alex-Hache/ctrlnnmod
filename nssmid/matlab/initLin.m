function BLA = initLin(u,y, nx, ts, model_type)

    switch model_type
        case 'discrete'
            dataLin = iddata(y,u,ts);
            opt = ssestOptions('EnforceStability', true,...
            'focus','simulation');
            % learn linear model on the full dataset
            linMod = ssest(dataLin,nx,opt,'Ts',ts, 'disturbanceModel','none','Feedthrough',0);
        case 'continuous'
            dataLin = iddata(y,u,ts);
            opt = ssestOptions('EnforceStability', true,...
            'focus','simulation');
            % learn linear model on the full dataset
            linMod = ssest(dataLin,nx, opt, 'disturbanceModel','none','Feedthrough',0);
        otherwise 
            error('Please specify if continuous or discrete time identification should be performed')
    end

    disp('Linear init. done.')

    [~,~,xCell] = sim(linMod,dataLin);
    sprintf("Fit percent : %d %%",linMod.Report.Fit.FitPercent)

    x = xCell;
    
    % normalize the linear model such that the states have a unit variance
    T = diag(std(x))^-1;
    BLA = struct();
    BLA.A = T*linMod.A*T^-1;
    BLA.B = T*linMod.B;
    BLA.C = linMod.C*T^-1;
    BLA.D = linMod.D;
end