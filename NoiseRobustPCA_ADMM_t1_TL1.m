function [L, S] = NoiseRobustPCA_ADMM_t1_TL1(M,Linear_Ti,a1,a2,n, lambda, mu,pho1,pho2, tol, max_iter,Lr,Sr,Li,Si)

    [m1, m2] = size(M);
    [n1,n2]=size(Li);
    unobserved = isnan(M);
    M(unobserved) = 0;
    normM = norm(M, 'fro');

    L=Li;
    S=Si;
    J=Li;

    R=Si;
    B=zeros(m1, m2);
    D=zeros(m1, m2);
    err_last=0;
    err_now=0;
    errt_L=1e10;
    errt_S=1e10;


    for iter = (1:max_iter)
        % ADMM step: update L and S
        % nuclear norm
        L=Do(lambda/pho1, J-B,a1);

        S=TL1min(mu/pho2,a2,R-D);
       
        J=((2/n)*(Linear_Ti).*(M-R)+pho1*(L+B))./((2/n)*Linear_Ti+pho1*ones(m1,m2));
        R=((2/n)*(Linear_Ti).*(M-J)+pho2*(S+D))./((2/n)*Linear_Ti+pho2*ones(m1,m2));
        B=B+L-J;
        D=D+S-R;

        errl=norm(L-Lr, 'fro')/norm(Lr, 'fro');
        errs=norm(S-Sr, 'fro')/norm(Sr, 'fro');

        errM=norm(M-Linear_Ti.*(L+S), 'fro')/norm(M, 'fro');
        %disp(iter);
        %disp(errM);
        %disp(errl);
        %disp(errs);
        
        if (iter == 1) || (mod(iter, 10) == 0) 
            err1=(1/n)*norm(M-Linear_Ti.*(L+S), 'fro')^2;
        
            %err2=lambda*nuclearnorm(L);
            [U1, St1, V1] = svd(L, 'econ');
            err2=lambda*TL1vec(diag(St1));    
            err3=mu*sum(abs(S(:)));
            err=err1+err2+err3;
            fprintf(1, 'iter: %04d\terr: %f\terr1: %f\t err2: %f\t err3: %f\n', ...
            iter, err,err1,err2, err3*1e8);
            if err2>=errt_L
                pho1=pho1*10;
            end
            if err3>=errt_S
                pho2=pho2*10;
            end
            %disp("pho");
            %disp(pho1);
            errt_L=err2;
            errt_S=err3;
            err_last=err_now;
            err_now=err;
            %fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
            fprintf(1, 'iter: %04d\terr: %f\terr1: %f\t err2: %f\t err3: %f\n', ...
            iter, err,err1,err2, err3*1e8);
            if (abs(err_last-err_now)/abs(err_now) < tol) break; end
        end
        
        %disp("err");
        %disp(err);
        %disp(err_last);
        %disp(err_now);
        %disp((abs(err_last-err_now)/abs(err_now)));
    end
end

function r = So(tau, X)
    % shrinkage operator
    disp("t2")
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = Do(tau, X,a)
    % shrinkage operator for singular values
    [U, S, V] = svd(X, 'econ');
    %L1 shrink
    disp("t1")
    %r = U*So(tau, S)*V';
    %disp(diag(S));
    %S=S/1;
    %TL1 shrink
    r = 1*U*TL1min(tau,a, S)*V';
end
function y=TL1function(x,a,lambda)
    y=sign(x) .* (2/3 * (a + abs(x)) .* cos(acos(1 - (27 * lambda * a * (a + 1)) ./ (2 * (a + abs(x)).^3))./3) - 2 * a ./ 3 + abs(x)./3);
end
function r=TL1min(lambda,a,X)
    t=0;
    if lambda <= (a^2) / (2 * (a + 1))
        t = lambda * (a + 1) / a;
    else
        % super-critical case
        t = sqrt(2 * lambda * (a + 1)) - a / 2;
    end
    r=X;
    r(abs(r)<=t)=0;
    index = abs(r) > t;
    r(index) = TL1function(r(index),a,lambda);
    
end

function n = nuclearnorm(X)
    s = svd(X);
    n = sum(s);
end




function v = shrinkTL1(s,lambda,a)
phi = acos(1-(0.5*27*lambda*a*(a+1))./(a+abs(s)).^3);

v = sign(s).*(2/3 * (a+abs(s)).* cos(phi/3) -2*a/3+abs(s)/3).*(abs(s)>lambda);
   
end

function XA= Linearmap(A,Linear,m1,m2)
[m, n] = size(A);
A_vec = reshape(A, [], 1);
B=Linear*A_vec;
XA= reshape(B,m1,m2);
end

function XA= Linearmap_diag(A,Linear,m1,m2)
[m, n] = size(A);
A_vec = reshape(A, [], 1);
B=Linear*A_vec;
XA= reshape(B,m1,m2);
end


function f=TL1vec(x)
    f=sum(2 * abs(x) ./ (1 + abs(x)));
end