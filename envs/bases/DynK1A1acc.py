import numpy as np

#==================================================================================================
class DynK1A1acc:
    
    #==============================================================================================
    def __init__(self, dt, tau_ax, tau_alpha, ini_x, ini_y, ini_yaw): # all float type
        # dt        = 0.01 [sec] : sampling time                                       -> 가속화 요구사항과 동역학 안정성을 고려해야 최적값을 찾을 예정
        # tau_ax    = 1.0  [-]   : time delay coefficient of longitudinal acceleration -> 종방향 가속도 차량 동특성을 반영하게 될 튜닝 계수
        # tau_alpha = 1.0  [-]   : time delay coefficient of angular acceleration      -> 회전   가속도 차량 동특성을 반영하게 될 튜닝 계수
        # ini_x     = X    [m]   : initial X-position in global coordinate (visual)    -> 영상엔진에서 전역좌표계 초기위치 X 
        # ini_y     = Y    [m]   : initial Y-position in global coordinate (visual)    -> 영상엔진에서 전역좌표계 초기위치 Y
        # ini_yaw   = yaw  [deg] : initial yaw angle in global coordinate (visual)     -> 영상엔진에서 전역좌표계 초기회전각도 yaw 
        self.dt = dt
        self.tau_ax = tau_ax
        self.tau_alpha = tau_alpha
        self.A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1/self.tau_ax, 0], [0, 0, 0, -1/self.tau_alpha]])
        self.B = np.array([[0, 0], [0, 0], [1/self.tau_ax, 0], [0, 1/self.tau_alpha]])
        self.X = np.zeros((4, ))
        self.x = ini_x
        self.y = ini_y
        self.yaw = ini_yaw
   
    #==============================================================================================
    def sim_once(self, ax_des, alpha_des): # all float type
        # ax_des    [m/s**2]   : desired logitudinal acceleration -> 기동/구동알고리즘에서 산출된 데이터
        # alpha_des [deg/s**2] : desired angular acceleration     -> 기동/구동알고리즘에서 산출된 데이터
        
        #----- formula
        # X = [vx, gamma(yaw_rate), ax, alpha(angular_acceleration)] 
        # X_dot = A * X + B * U

        #----- input
        U = np.array([ax_des, alpha_des])
        
        #----- computation
        X_dot = np.dot(self.A, self.X) + np.dot(self.B, U) 

        #----- numeric integration and coord. transformation
        self.X += X_dot * self.dt  # <-- ValueError: non-broadcastable output operand with shape (4,) doesn't match the broadcast shape (4,4)
        vx = self.X[0]
        yaw_rate = self.X[1]
        ax = self.X[2]
        alpha = self.X[3]
        self.x += vx * np.cos(np.radians(self.yaw)) * self.dt
        self.y += vx * np.sin(np.radians(self.yaw)) * self.dt
        self.yaw += yaw_rate * self.dt 
                
        #----- output (all float type)
        # x        [m]        : x-position in global coordinate
        # y        [m]        : y-position in global coordinate
        # yaw      [deg]      : yaw angle in global coordinate
        # vx       [m/s]      : vehicle volocity
        # yaw_rate [deg/s]    : yaw rate
        # ax       [m/s**2]   : logitudinal acceleration
        # alaph    [deg/s**2] : angular acceleration
        return self.x, self.y, self.yaw, vx, yaw_rate, ax, alpha
    
#==================================================================================================