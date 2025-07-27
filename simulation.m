   

    rng('default');
    %set parameters in the following order
    %size, rank, SR, noise,eg, lambda, mu,a1,a2 errL, errS,rank L,non-zero
    
    record = [
    300.0000    5.0000    0.2000    0   -5.0000   -6.0000    0    0    0.1067    0.4783   47.0000    0.0903    0;];
    para_number=size(record,1);
    % size rank SR noise errL errS rank L nonzeroS
    meanrecord=zeros(para_number,15);
    meanrecord(:, 1:4) = record(:, 1:4);
    minrecord=meanrecord;
    meanrecord_clean=meanrecord;
    maxiter=100;
    s=0.01;
    noiseLevel = 0.03;  %Change the noise level
    eg = 1;   %  for uniform setting, set eg = 1; for non-uniform setting, set eg = 2
    for para = 1:para_number
        a1=record(para,7);
        a2=record(para,8);
        lambda=10^record(para,5);
        mu=10^record(para,6);
        %errL errS rankL nonzeroS
        result50=zeros(maxiter,6);
        for iter=1:maxiter 
            rng('default');
            rng(iter);              
            d1_values = record(para,1);
            r_values  = record(para,2);
            SR_values = record(para,3);
            set = 0;
            SNR=20;
            % Preallocate results matrices
            num_models = length(d1_values) * length(r_values) * length(SR_values);
            Erra = zeros(1, num_models);
            Tima = zeros(1, num_models);
            result_err=zeros(1, maxiter);
            result_time=zeros(1, maxiter);
            % Iterate cross various combinations
            for d1 = d1_values
                
                d2 = d1;
                len = d1 * d2;
                for r = r_values
                    for SR = SR_values
                        set = set + 1;
                        Err = zeros(1, maxiter);
                        Tim = zeros(1, maxiter);

                        % Generate left and right matrix with rank r
                        L_left = randn(d1, r)/sqrt(r);
                        L_right = randn(d2, r)/sqrt(r);
                        L_0 = L_left * L_right';
                        S_0 = zeros(d1, d2);
                        idx_s = randperm(d1 * d2, round(s * d1*d2));
                        S_0(idx_s) = -1 + 2 * rand(1, length(idx_s));
                        if eg == 1
                            nzidx = randperm(len, round(SR * len));
                            nzidx=nzidx';
                            options.lamfactor = 1e-1; %needed to be tuned
                            options.sigma = 5e-3; 
                        elseif eg == 2      
                            pvec = ones(d1, 1);
                            cnt = round(0.1 * d1);
                            pvec(1:cnt) = 2 * pvec(1:cnt);
                            pvec(cnt + (1:cnt)) = 4 * pvec(cnt + (1:cnt));
                            pvec = d1 * pvec / sum(pvec);
                            qvec = ones(d2, 1);
                            cnt = round(0.1 * d2);
                            qvec(1:cnt) = 2 * qvec(1:cnt);
                            qvec(cnt + (1:cnt)) = 4 * qvec(cnt + (1:cnt));
                            qvec = d2 * qvec / sum(qvec);
                            probmatrix = rand(d1, d2) .* (pvec * qvec');
                            [~, sortidx] = sort(probmatrix(:), 'descend');
                            nzidx = sortidx(1:round(SR * len));
                            options.lamfactor = 1e-1; %needed to be tuned
                            options.sigma = 5e-3;
                        end      
                        % Main iteration loop
                        for trial = 1:1
                            % Generate observed matrix with noise or noiseless
                            Y = zeros(d1, d2);
                            W = zeros(d1, d2);
                            Mobs = zeros(d1, d2);
                            M=L_0+S_0;
                            Mobs = M(nzidx);
                            S_1= zeros(d1, d2);
                            S_1(nzidx)=S_0(nzidx);
                            S_0=S_1;
                            Ti=zeros(d1, d2);
                            Ti(nzidx)=1;
                            TiA=norm(Ti .* M, 'fro')^2;
                            randvec = randn(length(nzidx), 1);
                            Y(nzidx) = Mobs + (noiseLevel ) * randvec;
                            W(nzidx)=(noiseLevel ) * randvec;
        
                            indicator = zeros(d1*d2, 1); 
                            indicator(nzidx) = 1;   
                            %if linear map is diagonose
                            Linear_Ti=zeros(d1,d2);
                            Linear_Ti(nzidx)=1;
        
                            
                            % Call TL1 ADMM algorithm
                            t = cputime;
                            %disp("option");
                            %disp(options);
                            %disp(A);
                            %[M2, ~, ~] = TL1(A, options);
                            alpha=(1/3)*sqrt(d1*d2);
                            
                            tic;
                            %For nuclear norm: use function NoiseRobustPCA_ADMM_t1_nuclear
                            %For TL1 norm: use function
                            %NoiseRobustPCA_ADMM_t1_TL1 and the initial
                            %value is the result of nuclear norm
                            [L, S] = NoiseRobustPCA_ADMM_t1_nuclear(Y, Linear_Ti, a1,a2,SR*d1*d2 ,lambda,mu, 1e-7,1e-7, 1e-2, 100, L_0, S_0, Y, 0);
                            
                            elapsed_time = toc;
                            
                            errl = norm(L - L_0, 'fro') / norm(L_0, 'fro');
                            errs = norm(S - S_0, 'fro') / norm(S_0, 'fro');
                            total_err = errl + errs;
                            nonzeros=nnz(S) / numel(S);
                            %dice
                            S_bin = abs(S) > 0.1;
                            S0_bin = S_0 ~= 0;
                            intersection = nnz(S_bin & S0_bin);
                        
                            % non-zero element
                            total = nnz(S_bin) + nnz(S0_bin);
                            dice = 2 * intersection / total;    
                            max_singular = 0.01*svds(L, 1);
                            result50(iter, :) = [errl, errs,dice, rank(L,max_singular),nonzeros,elapsed_time]; 
                        end
                   end
                end
            end
        end

        average=mean(result50);
        meanrecord(para,5:10)=average;
        meanrecord(para,11)=std(result50(:, 1));
        meanrecord(para,12)=std(result50(:, 2));
        meanrecord(para,4)=noiseLevel;
        meanrecord(para,13)=std(result50(:, 3));
        meanrecord(para,14)=std(result50(:, 4));
        meanrecord(para,15)=std(result50(:, 5));


        % delete outlier
        errs100 = result50(:, 2);  

        Q1 = quantile(errs100, 0.25);
        Q3 = quantile(errs100, 0.75);
        IQR_val = Q3 - Q1;
        
        outlier_idx = (errs100 < Q1 - 1.5*IQR_val) | (errs100 > Q3 + 1.5*IQR_val);
        result50_clean = result50(~outlier_idx, :);
        average_clean=mean(result50_clean);
        meanrecord_clean(para,5:10)=average_clean;
        meanrecord_clean(para,11)=std(result50_clean(:, 1));
        meanrecord_clean(para,12)=std(result50_clean(:, 2));
        meanrecord_clean(para,4)=noiseLevel;
        meanrecord_clean(para,13)=std(result50_clean(:, 3));
        meanrecord_clean(para,14)=std(result50_clean(:, 4));
        meanrecord_clean(para,15)=std(result50_clean(:, 5));

    end
    fprintf('maxiter = %d, noiseLevel = %.2f, s = %.4f, eg = %.4f\n', maxiter, noiseLevel, s, eg);
    %display result
    meanrecord_clean(:,1)=1;
    disp('100 trials result: size, rank, SR, noise, errL, errS,diceS,rank L,non-zero,time, std_L,std_S,std_dice,std_rankL,std_non-zero');  
    disp(meanrecord_clean);
    





















