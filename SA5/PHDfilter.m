% Task: complete the code of the PHD filter class, which contains necessary functions to implement a Gaussian-mixture Probability Hypothesis Density (GM-PHD) filter:
%     1. PHD filter prediction;
%     2. PHD filter update;
%     3. extract object state estimates from (Gaussian mixture) Poisson intensity.
% For the first task (PHD filter prediction), your implementation should consist of the following steps:
%     1. Predict each Gaussian component in the Poisson intensity for pre-existing objects.
%     2. Add (Gaussian mixture) Poisson birth intensity to (Gaussian mixture) Poisson intensity for pre-existing objects.
% For the second task (PHD filter update), your implementation should consist of the following steps:
%     1. Construct update components resulted from missed detection.
%     2. Perform ellipsoidal gating for each Gaussian component in the Poisson intensity.
%     3. Construct update components resulted from object detections that are inside the gates. 
% There are a few alternatives when extracting object states from a Gaussian mixture multi-object density. In the third task (extract object state estimates), your implementation should consist of the following steps:
%     1. Get a mean estimate of the cardinality of objects by taking the summation of the weights of the Gaussian components  (rounded to the nearest integer), denoted as n. 
%     2. Extract n object states from the means of the n Gaussian components with the highest weights.
% Note: 
%     1. When calculating the weight of the new Gaussian component resulting from object detection, you should scale it properly by taking the clutter intensity into account.
%     2. In the estimator, you can use n = min(n, #Gaussian components) to make sure that your implementation still works if n > #Gaussian components.
%
% Note that obj.density is a function handle bound to MATLAB class GaussianDensity. For instance, you can simply call obj.density.update to perform a Kalman update instead of using your own code. 


classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the
    %PHD filter 
    %Model structures need to be called:
    %    sensormodel: a structure which specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure which specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure which specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array which specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture
            %                    components x 1) 
            
            obj.density = density_class_handle;
            obj.paras.w = [birthmodel.w]';
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
            Hkm1_km1 = length(obj.paras.w);
            Hk_km1 = Hkm1_km1 + length(birthmodel);
            w = zeros(Hk_km1, 1);
            w_old = obj.paras.w;
            states = struct('x',[],'P',[]);
            states_old = obj.paras.states;
            for h = 1 : length(birthmodel)
                w(h) = birthmodel(h).w;
                states(h, 1).x = birthmodel(h).x;
                states(h, 1).P = birthmodel(h).P;
            end

            for h = 1 : Hkm1_km1
                w(h+length(birthmodel)) = log(P_S) + w_old(h);
                state_pred = obj.density.predict(states_old(h), motionmodel);
                states(h+length(birthmodel), 1).x = state_pred.x;
                states(h+length(birthmodel), 1).P = state_pred.P;
            end

            obj.paras.w = w;
            obj.paras.states = states;

        end
        
        function obj = update(obj,z,measmodel,intensity_c,P_D,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: z: measurements --- matrix of size (measurement dimension 
            %          x number of measurements)
            %       intensity_c: Poisson clutter intensity --- scalar
            %       P_D: object detection probability --- scalar
            %       gating: a struct with two fields: P_G, size, used to
            %               specify the gating parameters
            
            Hk_km1 = length(obj.paras.w);
            num_measurements = size(z, 2);
            % first construct missed detection
            for i = 1 : Hk_km1 
                updated_weights(i, 1) = log(1-P_D) +  obj.paras.w(i);
                updated_states(i, 1) = obj.paras.states(i);
            end

            % gating 
            gate_indices = false(Hk_km1, num_measurements);
            for i = 1 : Hk_km1
                [~, gate_indices(i, :)] = obj.density.ellipsoidalGating(obj.paras.states(i), z, measmodel, gating.size);
            end

            h0 = Hk_km1; % starts from Hk_km1
            for h = 1 : Hk_km1
                for j = 1 : size(z, 2)
                    h0 = h0 + 1;
                    if gate_indices(h, j) == 1
                        updated_states(h0, 1) = obj.density.update(obj.paras.states(h), z(:, j), measmodel);
                        updated_weights(h0, 1) = log(P_D) + obj.paras.w(h) + obj.density.predictedLikelihood(obj.paras.states(h), z(:, j),measmodel);
                    else
                        updated_states(h0, 1) = struct('x', [], 'P', []);
                        updated_weights(h0, 1) = inf;
                    end
                end
            end



            for i = 1 : size(z, 2)
                if ~isempty(find(gate_indices(: ,i)))
                    subcomponent_indices = (find(gate_indices(: ,i)) - 1) * num_measurements + i + Hk_km1;
                    updated_weights(subcomponent_indices, :) = updated_weights(subcomponent_indices, :) - ...
                    log(intensity_c +  sum(exp(updated_weights(subcomponent_indices, :))));
                end
            end

            % extract real components
            idx = updated_weights~=inf;
            obj.paras.states = updated_states(idx);
            obj.paras.w = updated_weights(idx);
 

        end
        
        function obj = componentReduction(obj,reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            
            %Pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, reduction.w_min);
            %Merging
            if length(obj.paras.w) > 1
                [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, reduction.merging_threshold, obj.density);
            end
            %Capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, reduction.M);
        end
        
        function estimates = PHD_estimator(obj)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of
            %                  objects) 
            n = round(sum(exp(obj.paras.w)));
            [B,idx] = maxk(obj.paras.w, min(n, length(obj.paras.w)));

            estimates = [obj.paras.states(idx).x];
            assert(isequal(size(estimates, 2), length(idx)));

        end
        
    end
    
end
