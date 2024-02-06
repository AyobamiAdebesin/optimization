function Param_Est_SVIRD3_IRGN_Florida
% Experiments with real data for Florida

clear 
close all
clc
format long 
warning('off','all')

tic;

NumCurves = 100;
num_GN = 50;
tau0 = 1e8; 
power = 9; 
% factor = .1;

a = 1;
b = 140;
m = 10;

alpha = .8;

gammavd = 0.005/12.7/18.5;   
gammasd = 0.005/18.5;
gammasr = (1-0.005)/10;
gammavr = (1-0.005/12.7)/10;
p = 0.001021429;

delta1 = 1/90;

delta2 = 0;

N = 21589602; % Florida population

disp('RECOVERED EPIDEMIOLOGICAL PARAMETERS FROM LSQCURVEFIT')
disp('________________________________________________________________________________________________________________')
disp('j....... t1........ t2.........t3.........t4........ t5.........t6........t7.........t8.........t9.........t10..')
disp('________________________________________________________________________________________________________________')

load Florida.txt

tdata = Florida(:,1);
Idata = Florida(:,2);
Ddata = Florida(:,3);

lambdaI = 4e9;
lambdaD = 1e7;

n = length(tdata);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y0(1) = 11526646;
Y0(2) = 10039580;

Y0(3) = 12459;
Y0(4) = 10917;
Y0(5) = 0;
Y0(6) = 0;

theta0 = zeros(m,1);
theta0(1) = .5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Uncertainty quantification for the reconstructed transmission rate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preallocating for speed
curves_results_Incidence = zeros(n,NumCurves);
curves_results_Deaths = zeros(n,NumCurves); bt = zeros(n,NumCurves); 
repr = zeros(n,NumCurves);
Phatss = zeros(NumCurves,m); Rep_rate = zeros(NumCurves,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate the random numbers first for reproducibility
rng(0) % fixing the seed for reproducibility
tauI = repmat(abs(Idata),[1,NumCurves]);
curvesI = poissrnd(tauI);

tauD = repmat(abs(Ddata),[1,NumCurves]);
curvesD = poissrnd(tauD);

parfor iter = 1:NumCurves
theta = zeros(m,1);
theta(1) = .5;
    
ExpDataI = curvesI(:,iter);
ExpDataD = curvesD(:,iter);

options = optimoptions('lsqcurvefit','MaxIterations',0,...
        'Algorithm','levenberg-marquardt','InitDamping',1e5,'Disp','off'); 

for lp = 1:num_GN  
    
        % Evalaute the two Jacobians
        
        [~,~,~,~,~,~,I_prime] = lsqcurvefit(@(theta,tdata) I_function(theta,tdata,Y0,alpha,N,...
            gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b,lambdaI),theta,tdata,ExpDataI,[],[],options);
        
        [~,~,~,~,~,~,D_prime] = lsqcurvefit(@(theta,tdata) D_function(theta,tdata,Y0,alpha,N,...
            gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b,lambdaD),theta,tdata,ExpDataD,[],[],options);

        tau = tau0/lp^power;
%         tau = tau0/exp(factor*lp);
        
        I = I_function(theta,tdata,Y0,alpha,N,gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b,lambdaI);
        D = D_function(theta,tdata,Y0,alpha,N,gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b,lambdaD);
        P = eye(n) - (ExpDataI*ExpDataI')./(ExpDataI'*ExpDataI);
    
        step_theta = -(I_prime'*P*I_prime + D_prime'*D_prime + tau*eye(m))\...
            (I_prime'*P*I + D_prime'*(D - ExpDataD/lambdaD) + tau*(theta - theta0));

        theta = step_theta + theta; 
    
end

% I_prime'*P*I
% D_prime'*(D - ExpDataD/lambdaD)

repr(:,iter) = repr_function(theta,tdata,Y0,alpha,N,gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b);

Phatss(iter,:) = theta;  

I = lambdaI*I_function(theta,tdata,Y0,alpha,N,gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b,lambdaI);
D = lambdaD*D_function(theta,tdata,Y0,alpha,N,gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b,lambdaD);

Rep_rate(iter) = (Idata'*Idata)/(I'*Idata);

curves_results_Incidence(:,iter) = (Idata'*Idata)*I/(I'*Idata);
curves_results_Deaths(:,iter) = D;
bt(:,iter) = beta(tdata,theta,m,a,b);

%CREATION OF MATRIX FOR OUTPUT OF TABLE VALUES 
fprintf('%3d...%8.5f...%8.5f...%8.5f...%8.5f...%8.5f...%8.5f...%8.5f...%8.5f...%8.5f...%8.5f\n',...
    iter, theta(1), theta(2), theta(3), theta(4), theta(5), theta(6), theta(7), theta(8), theta(9), theta(10)); 

end

figure10 = figure; 
axes2 = axes('Parent',figure10,... 
    'AmbientLightColor',[0.941176470588235 0.941176470588235 0.941176470588235]);
box(axes2,'on');
hold(axes2,'all')

param_1=[mean(Phatss(:,1)) plims(Phatss(:,1),0.025) plims(Phatss(:,1),0.975)];

param_2=[mean(Phatss(:,2)) plims(Phatss(:,2),0.025) plims(Phatss(:,2),0.975)];

param_3=[mean(Phatss(:,3)) plims(Phatss(:,3),0.025) plims(Phatss(:,3),0.975)];

param_4=[mean(Phatss(:,4)) plims(Phatss(:,4),0.025) plims(Phatss(:,4),0.975)];

param_5=[mean(Phatss(:,5)) plims(Phatss(:,5),0.025) plims(Phatss(:,5),0.975)];

param_6=[mean(Phatss(:,6)) plims(Phatss(:,6),0.025) plims(Phatss(:,6),0.975)];

param_7=[mean(Phatss(:,7)) plims(Phatss(:,7),0.025) plims(Phatss(:,7),0.975)];

param_8=[mean(Phatss(:,8)) plims(Phatss(:,8),0.025) plims(Phatss(:,8),0.975)];

param_9=[mean(Phatss(:,9)) plims(Phatss(:,9),0.025) plims(Phatss(:,9),0.975)];

param_10=[mean(Phatss(:,10)) plims(Phatss(:,10),0.025) plims(Phatss(:,10),0.975)];

param_11=[mean(Rep_rate(:)) plims(Rep_rate(:),0.025) plims(Rep_rate(:),0.975)];


cad1=strcat('\theta_1=',num2str(param_1(end,1),2),'(95%CI:[',num2str(param_1(end,2),2),',',num2str(param_1(end,3),2),'])');
cad2=strcat('\theta_2=',num2str(param_2(end,1),2),'(95%CI:[',num2str(param_2(end,2),2),',',num2str(param_2(end,3),2),'])');
cad3=strcat('\theta_3=',num2str(param_3(end,1),2),'(95%CI:[',num2str(param_3(end,2),2),',',num2str(param_3(end,3),2),'])');
cad4=strcat('\theta_4=',num2str(param_4(end,1),2),'(95%CI:[',num2str(param_4(end,2),2),',',num2str(param_4(end,3),2),'])');
cad5=strcat('\theta_5=',num2str(param_5(end,1),2),'(95%CI:[',num2str(param_5(end,2),2),',',num2str(param_5(end,3),2),'])');

cad6=strcat('\theta_6=',num2str(param_6(end,1),2),'(95%CI:[',num2str(param_6(end,2),2),',',num2str(param_6(end,3),2),'])');
cad7=strcat('\theta_7=',num2str(param_7(end,1),2),'(95%CI:[',num2str(param_7(end,2),2),',',num2str(param_7(end,3),2),'])');
cad8=strcat('\theta_8=',num2str(param_8(end,1),2),'(95%CI:[',num2str(param_8(end,2),2),',',num2str(param_8(end,3),2),'])');
cad9=strcat('\theta_9=',num2str(param_9(end,1),2),'(95%CI:[',num2str(param_9(end,2),2),',',num2str(param_9(end,3),2),'])');
cad10=strcat('\theta_{10}=',num2str(param_10(end,1),2),'(95%CI:[',num2str(param_10(end,2),2),',',num2str(param_10(end,3),2),'])');
cad11=strcat('\psi=',num2str(param_11(end,1),3),'(95%CI:[',num2str(param_11(end,2),3),',',num2str(param_11(end,3),3),'])');

% suptitle(strcat('\fontsize{12}',cad1,';',cad2,';',cad3,';',cad4, ';',cad5));

subplot(2,5,1)
histogram(Phatss(:,1),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_1')
subplot(2,5,2)
histogram(Phatss(:,2),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_2')
subplot(2,5,3)
histogram(Phatss(:,3),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_3')
subplot(2,5,4)
histogram(Phatss(:,4),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_4')
subplot(2,5,5)
histogram(Phatss(:,5),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_5')

% suptitle(strcat('\fontsize{12}',cad6,';',cad7,';',cad8,';',cad9, ';',cad10));
subplot(2,5,6)
histogram(Phatss(:,6),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_6')
subplot(2,5,7)
histogram(Phatss(:,7),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_7')
subplot(2,5,8)
histogram(Phatss(:,8),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_8')
subplot(2,5,9)
histogram(Phatss(:,9),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_9')
subplot(2,5,10)
histogram(Phatss(:,10),'FaceColor',	[0.9290 0.6940 0.1250])
xlabel('\fontsize{16}\theta_{10}')

figure(figure10) 

disp([cad1 cad2 cad3 cad4 cad5])
disp([cad6 cad7 cad8 cad9 cad10])
disp(cad11)

tcalendar = datetime(2021,07,09) + caldays(1:length(tdata));

figure1A = figure;
histogram(Rep_rate(:),'FaceColor',	[1.000000 0.800000 0.640000])
xlabel('\fontsize{16} Reporting Rate')
figure(figure1A) 


figure1 = figure;

line1 = plot(tcalendar,curves_results_Incidence,'Color', '[0.3010 0.7450 0.9330]');
set(line1,'LineWidth',2)
hold on 

line3 = plot(tcalendar,Idata,'bo');
set(line3,'LineWidth',2)
hold on

line4 = plot(tcalendar,mean(curves_results_Incidence,2),'-k');
set(line4,'LineWidth',2)

xlim([datetime(2021,07,9) datetime(2021,11,25)])

legend([line1(1) line3(1) line4(1)],{ 'Reconstructed Incidence Curves',...
     'Reported Incidence Data', 'Mean of Reconstructed Incidence Curves',},'FontSize',14,'Location','best');

xlabel('\fontsize{18}Time (days)');
ylabel('\fontsize{18}Case Incidence')

figure(figure1) 

figure2 = figure;

line1a = plot(tcalendar,curves_results_Deaths,'Color', '[1 .7 0.8]');
set(line1a,'LineWidth',2)
hold on 

line3a = plot(tcalendar,Ddata,'mo');
set(line3a,'LineWidth',2)
hold on

line4a = plot(tcalendar,mean(curves_results_Deaths,2),'-k');
set(line4a,'LineWidth',2)

xlim([datetime(2021,07,9) datetime(2021,11,25)])

legend([line1a(1) line3a(1) line4a(1)],{ 'Reconstructed Daily Deaths',...
     'Reported Data on Daily Deaths', 'Mean of Reconstructed Daily Deaths',},'FontSize',14,'Location','best');

xlabel('\fontsize{18}Time (days)');
ylabel('\fontsize{18}Daily Deaths')

figure(figure2) 

figure6 = figure;

line10 = plot(tcalendar, bt,'Color', '[0.4660 0.6740 0.1880]');
set(line10,'Linewidth',2)
hold on
line2a = plot(tcalendar,mean(bt,2),'-k');
set(line2a,'LineWidth',2)

hold on

line5 = plot(tcalendar, beta(tdata,theta0,m,a,b),'--c');
set(line5,'LineWidth',2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

line5r = line([datetime(2021,07,9) datetime(2021,11,25)], [1 1],'linestyle', '-.','Color',[0.6 0.3 0.3] );
set(line5r,'LineWidth',3)
hold on

line10r = plot(tcalendar, repr,'g');
set(line10r,'Linewidth',2)
hold on

line2r = plot(tcalendar,mean(repr,2),'-k');
set(line2r,'LineWidth',2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ylim([0.0 3.5])
xlim([datetime(2021,07,9) datetime(2021,11,25)])

legend([line10r(1) line2r(1)  line10(1) line2a(1) line5(1) ],...
    {'Reconstructed Effective Reproduction','Mean of Reconstructed Effective Reproduction',...
    'Reconstructed Transmission Rate','Mean of Reconstructed Transmission Rate',...
     'Initial Approximation'},'FontSize',14,'Location','best');
% legend('Effective Reproduction Number','Initial Guess','best');
xlabel('\fontsize{18}Time (days)');
ylabel('$\mathcal{R}_e(t)\,\,\mbox{and}\,\,\beta(t)$','FontSize',18,'Interpreter','latex')
figure(figure6)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('RECOVERED REPORTING RATE FOR INCIDENCE CASES')
fprintf('%8.5f\n', mean(Rep_rate));

% fid = fopen('Rep_Rate_Combined.txt','a');
% fprintf(fid,'%8.5f %s\n', mean(Rep_rate), 'Florida_rep_rate');
% fclose(fid);

toc;
end

% FUNCTION DEFINITIONS

function dydt = svird(t,y,theta,N,alpha,p,gammasd,gammasr,gammavr,gammavd,delta1,delta2,m,a,b)

dydt = zeros(6,1);
dydt(1) = -beta(t,theta,m,a,b).*y(1).*(y(3)+y(4))./(N - y(6)) - p.*y(1) + delta1*y(5) + delta2*y(2);
dydt(2) = p.*y(1) - (1-alpha)*beta(t,theta,m,a,b).*y(2).*(y(3)+y(4))./(N - y(6)) - delta2*y(2);
dydt(3) = beta(t,theta,m,a,b).*y(1).*(y(3)+y(4))./(N - y(6)) - (gammasr + gammasd)*y(3);
dydt(4) = (1-alpha).*beta(t,theta,m,a,b).*y(2).*(y(3)+y(4))./(N - y(6)) - (gammavr + gammavd)*y(4);
dydt(5) = gammasr.*y(3) + gammavr.*y(4) - delta1*y(5);
dydt(6) = gammasd.*y(3) + gammavd.*y(4);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yy = beta(t,theta,m,a,b) 

       yy = 0;
   for j = 1:m
       yy = yy + theta(j).*leg(j-1,t,a,b);
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function P = leg(j,t,a,b)

x = (2.*t - a - b)./(b - a);

if j == 0
   P1 = 1; P = P1; 
elseif j == 1 
   P2 = x; P = P2; 
else   
   P1 = 1; P2 = x; 
       for k = 2:j
           P3 = ((2*(k-1)+1).*x.*P2 - (k-1).*P1)./k;
           P1 = P2; P2 = P3;   
       end
           P = P3;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q = I_function(theta,tdata,Y0,alpha,N,gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b,lambdaI)

[~,Y] = ode23s(@(t,y) svird(t,y,theta,N,alpha,p,gammasd,gammasr,gammavr,gammavd,delta1,delta2,m,a,b),tdata,Y0);

q = beta(tdata,theta,m,a,b).*Y(:,1).*(Y(:,3)+Y(:,4))./(N - Y(:,6)) +...
    (1-alpha)*beta(tdata,theta,m,a,b).*Y(:,2).*(Y(:,3)+Y(:,4))./(N - Y(:,6));
q = q/lambdaI;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q = D_function(theta,tdata,Y0,alpha,N,gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b,lambdaD)

[~,Y] = ode23s(@(t,y) svird(t,y,theta,N,alpha,p,gammasd,gammasr,gammavr,gammavd,delta1,delta2,m,a,b),tdata,Y0);

q = gammasd.*Y(:,3) + gammavd.*Y(:,4);
q = q/lambdaD;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q = repr_function(theta,tdata,Y0,alpha,N,gammasd,gammavd,p,gammasr,gammavr,delta1,delta2,m,a,b)

[~,Y] = ode23s(@(t,y) svird(t,y,theta,N,alpha,p,gammasd,gammasr,gammavr,gammavd,delta1,delta2,m,a,b),tdata,Y0);

q = beta(tdata,theta,m,a,b).*Y(:,1)./((N-Y(:,6)).*(gammasr + gammasd))...
          + (1 - alpha).*beta(tdata,theta,m,a,b).*Y(:,2)./((N-Y(:,6)).*(gammavr + gammavd));

end


