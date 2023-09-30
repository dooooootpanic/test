import torch
import pandas as pd
import numpy as np
from torch.autograd import grad
from torch.utils.data import TensorDataset, random_split, DataLoader


# 读取CSV文件
device=torch.device("cuda"if torch.cuda.is_available() else"cpu")
df = pd.read_csv(r"C:\Users\亓宇鹏\Documents\Tencent Files\1134944482\FileRecv\数据集929.csv")


# 假设 df 是你的原始数据
# 初始化一个空的DataFrame，用于存储重塑后的数据
reshaped_df = pd.DataFrame()

# 提取前3列
first_three_columns = df.iloc[:, :3]
first_14_columns = df.iloc[:, :14]
# 保存第一组的列名
first_14_column_names = first_14_columns.columns.tolist()

# 按照每11列为一组进行重塑，从第14列开始
num_columns = df.shape[1]
num_rows = df.shape[0]
first_group = True
for i in range(14, num_columns, 11):
    # 提取当前组的列
    group_df1 = df.iloc[:, i:i + 11]

    # 将前3列添加到当前组的前面
    group_df = pd.concat([first_three_columns, group_df1], axis=1)

    # 对于除第一组之外的其他组，去掉表头

    group_df.columns = range(group_df.shape[1])


    # 将当前组的列堆叠到第一组的下面
    reshaped_df = pd.concat([reshaped_df, group_df], ignore_index=True)
# 将第一组的列名应用到整个DataFrame上
reshaped_df.columns = first_14_column_names + reshaped_df.columns[len(first_14_column_names):].tolist()

reshaped_df=pd.concat([first_14_columns, reshaped_df]).reset_index(drop=True)


# 重塑后的数据行数应该是原始数据行数的倍数
num_groups = (num_columns - 14) // 11 + 1
reshaped_df = reshaped_df.iloc[:num_rows * num_groups, :]

# 保存重塑后的数据到一个新的CSV文件
reshaped_df.to_csv('reshaped_data.csv', index=False)
print(reshaped_df)

# 假设前4列是输入数据，第5列是标签
qiansilie = torch.tensor(reshaped_df.iloc[:, :4].values, dtype=torch.float32)  # 使用适当的数据类型
diwulie = torch.tensor(reshaped_df.iloc[:, 4].values, dtype=torch.long)  # 使用适当的数据类型

# 创建一个TensorDataset
dataset = TensorDataset(qiansilie, diwulie)

# 定义训练集和验证集的大小
train_size = int(0.8 * len(dataset))  # 假设训练集是数据集的80%
val_size = len(dataset) - train_size

# 使用random_split分割数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


# 定义芯片的几何形状
pcb = {'x': [-0.08, 0.08], 'y': [-0.08, 0.08], 'z': [0, 0.0065]}  # pcb板
adhesive = {'x': [-0.04, 0.04], 'y': [-0.04, 0.04], 'z': [0.0065, 0.007]}  # 粘合剂层
chip = {'x': [-0.04, 0.04], 'y': [-0.04,0.04], 'z': [0.07, 0.0135]}  # 芯片

# 定义热交换系数和空气温度
h = 20
T_air = 293.15

# 定义热源
def heat_source(x, y, z, t):
    condition_chip = (chip['x'][0] <= x) & (x <= chip['x'][1]) & (chip['y'][0] <= y) & (y <= chip['y'][1]) & (chip['z'][0] <= z) & (z <= chip['z'][1])

    source = torch.where(condition_chip, 2.404e7, 0)
    return source


# 定义热导率函数
def get_k(x, y, z, k_pcb, k_adhesive, k_chip):
    condition_pcb = (pcb['x'][0] <= x) & (x <= pcb['x'][1]) & (pcb['y'][0] <= y) & (y <= pcb['y'][1]) & (pcb['z'][0] <= z) & (z <= pcb['z'][1])
    condition_adhesive = (adhesive['x'][0] <= x) & (x <= adhesive['x'][1]) & (adhesive['y'][0] <= y) & (y <= adhesive['y'][1]) & (adhesive['z'][0] <= z) & (z <= adhesive['z'][1])
    condition_chip = (chip['x'][0] <= x) & (x <= chip['x'][1]) & (chip['y'][0] <= y) & (y <= chip['y'][1]) & (chip['z'][0] <= z) & (z <= chip['z'][1])

    k = torch.where(condition_pcb, k_pcb, torch.where(condition_adhesive, k_adhesive, torch.where(condition_chip, k_chip, 0)))
    return k
  # 如果不在任何定义的区域内，我们可以假设 k = 0

# 定义密度和比热容
def get_rho_and_c(x, y, z, rho_pcb, rho_adhesive, rho_chip, c_pcb, c_adhesive, c_chip):
    condition_pcb = (pcb['x'][0] <= x) & (x <= pcb['x'][1]) & (pcb['y'][0] <= y) & (y <= pcb['y'][1]) & (pcb['z'][0] <= z) & (z <= pcb['z'][1])
    condition_adhesive = (adhesive['x'][0] <= x) & (x <= adhesive['x'][1]) & (adhesive['y'][0] <= y) & (y <= adhesive['y'][1]) & (adhesive['z'][0] <= z) & (z <= adhesive['z'][1])
    condition_chip = (chip['x'][0] <= x) & (x <= chip['x'][1]) & (chip['y'][0] <= y) & (y <= chip['y'][1]) & (chip['z'][0] <= z) & (z <= chip['z'][1])

    rho = torch.where(condition_pcb, rho_pcb, torch.where(condition_adhesive, rho_adhesive, torch.where(condition_chip, rho_chip, 0)))
    c = torch.where(condition_pcb, c_pcb, torch.where(condition_adhesive, c_adhesive, torch.where(condition_chip, c_chip, 0)))

    return rho, c
  # 如果不在任何定义的区域内，我们可以假设rho = 0, c = 0
#定义
rho_pcb=1900
rho_adhesive=1673
rho_chip=2329
c_pcb=1369
c_adhesive=1000
c_chip=700
# 热传导系数
k_pcb = 0.3
k_adhesive = 2.5
k_chip = 80
# 这是温度输入模型
class TemperatureModel(torch.nn.Module):
    def __init__(self):
        super(TemperatureModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1)
        )

    def forward(self, x, create_graph=True):
        return self.net(x)


# 创建模型实例并移动到GPU
model = TemperatureModel().to(device)


# 定义温度函数
def Tmodel(x, y, z, t, model,create_graph=False):
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    t.requires_grad_(True)
    T = model(torch.stack([x, y, z, t], dim=-1).to(device), create_graph=True)  # 使用模型预测温度

    grad_outputs = torch.ones_like(T)
    T_x = grad(T, x, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]  # 计算T对x的导数
    T_y = grad(T, y, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]  # 计算T对y的导数
    T_z = grad(T, z, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]  # 计算T对z的导数
    T_t = grad(T, t, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]  # 计算T对z的导数

    return T, T_x, T_y, T_z, T_t



# 定义热传导方程
def heat_equation(x, y, z, t, model, k_pcb, k_adhesive, k_chip, rho_pcb, rho_adhesive, rho_chip, c_pcb, c_adhesive,
                  c_chip):
    k = get_k(x, y, z, k_pcb, k_adhesive, k_chip)
    rho, c = get_rho_and_c(x, y, z, rho_pcb, rho_adhesive, rho_chip, c_pcb, c_adhesive, c_chip)
    T, T_x, T_y, T_z,T_t = Tmodel(x, y, z, t, model)

    grad_outputs = torch.ones_like(T).squeeze()

    T_xx = grad(T_x, x, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    T_yy = grad(T_y, y, grad_outputs=grad_outputs, create_graph=True,allow_unused=True)[0]
    T_zz = grad(T_z, z, grad_outputs=grad_outputs, create_graph=True,allow_unused=True)[0]

    f = rho * c * T_t - (k * T_xx + k * T_yy + k * T_zz) - heat_source(x, y, z, t)  # 热传导方程
    return f


# 定义边界条件
def boundary_conditions(x, y, z, t, model):
    k = get_k(x, y, z, k_pcb, k_adhesive, k_chip)  # 获取相应的热导率
    # 上表面与空气进行热交换
    z_top = torch.full_like(x, chip['z'][1]).requires_grad_(True)
    T, _, _, T_z, _ = Tmodel(x, y, z_top, t, model)
    conditions = [h * (T - T_air) - k * T_z]

    # 下表面是绝热的
    z_bottom = torch.full_like(x, chip['z'][0]).requires_grad_(True)
    T, _, _, T_z, _ = Tmodel(x, y, z_bottom, t, model)

    conditions.append(h * (T - T_air) - k * T_z)

    for boundary in [pcb, adhesive, chip]:
        for i in range(2):
            x_temp = torch.full_like(x,boundary['x'][i],requires_grad=True )
            T, T_x, _, _ , _ = Tmodel(x_temp, y, z, t, model)
            conditions.append(h * (T - T_air) - k * T_x)
            y_temp = torch.full_like(x,boundary['y'][i], requires_grad=True )
            T, _, T_y, _, _  = Tmodel(x, y_temp, z, t, model)
            conditions.append(h * (T - T_air) - k * T_y)

    return conditions


# 定义温度初始条件
def initial_conditions(x, y, z, t, model):
    T, _, _, _ ,_= Tmodel(x, y, z, torch.zeros_like(t).to(device), model)
    return T - 293.15


# 定义损失函数
def losstem(x, y, z, t,  T_actual, Tmodel, k_pcb, k_adhesive, k_chip, rho_pcb, rho_adhesive, rho_chip, c_pcb, c_adhesive, c_chip,create_graph=True):
    f = heat_equation(x, y, z, t, model, k_pcb, k_adhesive, k_chip, rho_pcb, rho_adhesive, rho_chip, c_pcb, c_adhesive, c_chip)
    conditions = boundary_conditions(x, y, z, t, model)
    ic = initial_conditions(x, y, z, t,model)
    T, _, _, _ , _= Tmodel(x, y, z, t, model, create_graph=True)
    loss = torch.mean(f ** 2) + torch.mean(torch.stack(conditions) ** 2) + torch.mean(ic ** 2) + torch.mean(
        (T - T_actual) ** 2)#包含了方程 边界条件 初始调价 data条件
    return loss








    # 在这一步，你的 boundary_conditions 函数应该在一批数据上进行计算，并将结果保存下来。
    # 例如，你可以将结果添加到一个列表中，或者直接写入文件，以便后续处理。
    # 注意，你可能需要修改 boundary_conditions 函数，使其能够处理批量数据。

######################################################################################


class DisplacementModel(torch.nn.Module):
    def __init__(self):
        super(DisplacementModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 50),#位移的输入应该是x y z T
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 3)
        )

    def forward(self, x):
        return self.net(x)

# 创建位移模型实例并移动到GPU
displacement_model = DisplacementModel().to(device)

# 热膨胀系数
alpha_chip = 2.6E-6
alpha_adhesive = 22E-6
alpha_pcb = 18E-6

# 弹性模量和泊松比
E_pcb = 22000
mu_pcb = 0.15
E_adhesive = 7789
mu_adhesive = 0.311
E_chip = 106600
mu_chip = 0.25



# 定义热膨胀系数函数
def get_alpha(x, y, z, alpha_pcb, alpha_adhesive, alpha_chip):
    condition_pcb = (pcb['x'][0] <= x) & (x <= pcb['x'][1]) & (pcb['y'][0] <= y) & (y <= pcb['y'][1]) & (pcb['z'][0] <= z) & (z <= pcb['z'][1])
    condition_adhesive = (adhesive['x'][0] <= x) & (x <= adhesive['x'][1]) & (adhesive['y'][0] <= y) & (y <= adhesive['y'][1]) & (adhesive['z'][0] <= z) & (z <= adhesive['z'][1])
    condition_chip = (chip['x'][0] <= x) & (x <= chip['x'][1]) & (chip['y'][0] <= y) & (y <= chip['y'][1]) & (chip['z'][0] <= z) & (z <= chip['z'][1])

    alpha = torch.where(condition_pcb, alpha_pcb, torch.where(condition_adhesive, alpha_adhesive, torch.where(condition_chip, alpha_chip, 0)))

    return alpha
  # 如果不在任何定义的区域内，我们可以假设 alpha = 0





# 定义弹性模量和泊松比函数
def get_E_and_mu(x, y, z, E_pcb, E_adhesive, E_chip, mu_pcb, mu_adhesive, mu_chip):
    condition_pcb = (pcb['x'][0] <= x) & (x <= pcb['x'][1]) & (pcb['y'][0] <= y) & (y <= pcb['y'][1]) & (pcb['z'][0] <= z) & (z <= pcb['z'][1])
    condition_adhesive = (adhesive['x'][0] <= x) & (x <= adhesive['x'][1]) & (adhesive['y'][0] <= y) & (y <= adhesive['y'][1]) & (adhesive['z'][0] <= z) & (z <= adhesive['z'][1])
    condition_chip = (chip['x'][0] <= x) & (x <= chip['x'][1]) & (chip['y'][0] <= y) & (y <= chip['y'][1]) & (chip['z'][0] <= z) & (z <= chip['z'][1])

    E = torch.where(condition_pcb, E_pcb, torch.where(condition_adhesive, E_adhesive, torch.where(condition_chip, E_chip, 0)))
    mu = torch.where(condition_pcb, mu_pcb, torch.where(condition_adhesive, mu_adhesive, torch.where(condition_chip, mu_chip, 0)))

    return E, mu
 # 如果不在任何定义的区域内，我们可以假设 E = 0, mu = 0



class DisplacementModel(torch.nn.Module):
    def __init__(self):
        super(DisplacementModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 3)
        )

    def forward(self, x):
        return self.net(x)

model_u = DisplacementModel().to(device)

# 定义位移函数
# 定义位移函数
def displacement(x, y, z, t, model_u, model):  # 注意这里添加了 model 作为参数

    # 在这里调用 Tmodel 函数来获取 T_value
    T_value, _, _, _, _ = Tmodel(x, y, z, t, model)  # 假设 Tmodel 返回四个值，我们只取第一个

    # 确保 T_value 与其他变量具有相同的形状
    T_value = T_value.squeeze(-1)

    # 将 x, y, z, t, T_value 组合成一个张量
    inputs = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1), t.unsqueeze(-1), T_value.unsqueeze(-1)), dim=-1)

    # 然后直接将这个张量传递给 model_u
    disp = model_u(inputs)
    u, v, w = disp[:, 0], disp[:, 1], disp[:, 2]
    # 假设 u, v, w 已经被计算出来了
    grad_outputs = torch.ones_like(u)

    # 计算 u, v, w 关于 x, y, z 的梯度
    u_x = grad(u, x, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    u_y = grad(u, y, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    u_z = grad(u, z, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]

    grad_outputs = torch.ones_like(v)
    v_x = grad(v, x, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    v_y = grad(v, y, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    v_z = grad(v, z, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]

    grad_outputs = torch.ones_like(w)
    w_x = grad(w, x, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    w_y = grad(w, y, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    w_z = grad(w, z, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]

    # 现在你有了 u, v, w 关于 x, y, z 的梯度

    return u, u_x, u_y, u_z, v, v_x, v_y, v_z, w, w_x, w_y, w_z





def displaceloss(x, y, z, t, model_u, mu, alpha, model):
    ###定义了二阶导
    u, u_x, u_y, u_z, v, v_x, v_y, v_z, w, w_x, w_y, w_z = displacement(x, y, z, t, model_u, model)
    grad_outputs_u_x = torch.ones_like(u_x)
    u_xx = grad(u_x, x, grad_outputs=grad_outputs_u_x, create_graph=True, allow_unused=True)[0]

    grad_outputs_u_y = torch.ones_like(u_y)
    u_yy = grad(u_y, y, grad_outputs=grad_outputs_u_y, create_graph=True, allow_unused=True)[0]

    grad_outputs_u_z = torch.ones_like(u_z)
    u_zz = grad(u_z, z, grad_outputs=grad_outputs_u_z, create_graph=True, allow_unused=True)[0]

    grad_outputs_v_x = torch.ones_like(v_x)
    v_xx = grad(v_x, x, grad_outputs=grad_outputs_v_x, create_graph=True, allow_unused=True)[0]

    grad_outputs_v_y = torch.ones_like(v_y)
    v_yy = grad(v_y, y, grad_outputs=grad_outputs_v_y, create_graph=True, allow_unused=True)[0]

    grad_outputs_v_z = torch.ones_like(v_z)
    v_zz = grad(v_z, z, grad_outputs=grad_outputs_v_z, create_graph=True, allow_unused=True)[0]

    grad_outputs_w_x = torch.ones_like(w_x)
    w_xx = grad(w_x, x, grad_outputs=grad_outputs_w_x, create_graph=True, allow_unused=True)[0]

    grad_outputs_w_y = torch.ones_like(w_y)
    w_yy = grad(w_y, y, grad_outputs=grad_outputs_w_y, create_graph=True, allow_unused=True)[0]

    grad_outputs_w_z = torch.ones_like(w_z)
    w_zz = grad(w_z, z, grad_outputs=grad_outputs_w_z, create_graph=True, allow_unused=True)[0]

    #前面已经定义过一次T的导数了 这里应该不用定义了
    T, T_x, T_y, T_z ,T_t= Tmodel(x, y, z, t, model)

    pde_u = u_xx + (1 - mu) / 2 * (u_yy + u_zz) + (1 + mu) / 2 * (v_x * v_y + w_x * w_z) + (1 + mu) * alpha * T_x
    pde_v = v_yy + (1 - mu) / 2 * (v_xx + v_zz) + (1 + mu) / 2 * (u_x * u_y + w_y * w_z) +(1 + mu) * alpha * T_y
    pde_w = w_zz + (1 - mu) / 2 * (w_xx + w_yy) + (1 + mu) / 2 * (u_x * u_z + v_y * v_z) + (1 + mu) * alpha * T_z

    return torch.mean(pde_u ** 2) + torch.mean(pde_v ** 2) + torch.mean(pde_w ** 2)#所有的方程loss


# 定义边界条件损失函数
def boundary_loss(x, y, z, t, model_u):
    u, *_ = displacement(x, y, z, torch.zeros_like(t), model_u,model)
    return torch.mean(u ** 2)
T_ref = 293.15

# 在损失函数中添加一个新的项
def ref_temp_loss(x, y, z, t, model_u, Tmodel):
    u,*_ = displacement(x, y, z, torch.full_like(t, T_ref), model_u,model)
    T,*_ = Tmodel(x, y, z, torch.full_like(t, T_ref), model)
    return torch.mean(u ** 2) + torch.mean((T - T_ref) ** 2)

def compute_stress(x, y, z, t, T,model_u):
    u, u_x, u_y, u_z, v, v_x, v_y, v_z, w, w_x, w_y, w_z = displacement(x, y, z, t, model_u,model)
    E, mu = get_E_and_mu(x, y, z)
    alpha = get_alpha(x, y, z)

    # 计算应力分量
    common_factor = E / (1 - mu ** 2)
    thermal_factor = E * alpha * T * (1 - mu) / (1 - mu ** 2)

    sigma_xx = common_factor * (u_x + mu * v_y + mu * w_z) - thermal_factor
    sigma_yy = common_factor * (v_y + mu * u_x + mu * w_z) - thermal_factor
    sigma_zz = common_factor * (w_z + mu * u_x + mu * v_y) - thermal_factor

    sigma_xy = E / (2 * (1 + mu)) * (u_y + v_x)
    sigma_xz = E / (2 * (1 + mu)) * (u_z + w_x)
    sigma_yz = E / (2 * (1 + mu)) * (v_z + w_y)



    return x, y, z, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz


# 加载实际的热应力数据
sxx_actual = torch.from_numpy(data['sxx'].values.astype(np.float32)).to(device)

# 加载数据
data = pd.read_csv('reshaped_data.csv')
print(data.columns)

x = torch.from_numpy(data['x'].values.astype(np.float32)).to(device)
y = torch.from_numpy(data['y'].values.astype(np.float32)).to(device)
z = torch.from_numpy(data['z'].values.astype(np.float32)).to(device)
t = torch.from_numpy(data['t'].values.astype(np.float32)).to(device)
T_actual = torch.from_numpy(data['T'].values.astype(np.float32)).to(device)  # 加载实际的温度数据


batch_size = 100

# 计算批次数量
num_batches = len(x) // batch_size
# 假设 x, y, z, t, T_actual 已经是 Tensor 类型并且已经加载到了设备上
# 如果不是，您可以使用 torch.from_numpy 来转换它们

# 将数据合并到一个 TensorDataset 中
dataset = TensorDataset(x, y, z, t, T_actual)

# 定义批次大小
batch_size = 64

# 使用 random_split 将数据集分为训练集和验证集
train_size = int(0.8 * len(dataset))  # 计算训练集的大小
val_size = len(dataset) - train_size  # 计算验证集的大小
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# 计算批次数量
num_batches = len(x) // batch_size
# 创建一个新的优化器，包含热应力模型的参数
optimizer = torch.optim.Adam(list(model.parameters()) + list(model_u.parameters()))

# 训练模型
# 对每个批次进行处理
num_epochs = 10  # 设置你想要的epoch数量
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # 训练阶段
    for x_batch, y_batch, z_batch, t_batch, T_actual_batch in train_loader:
        x_batch, y_batch, z_batch, t_batch, T_actual_batch = (
            x_batch.to(device),
            y_batch.to(device),
            z_batch.to(device),
            t_batch.to(device),
            T_actual_batch.to(device),
        )

        optimizer.zero_grad()
        E, mu = get_E_and_mu(x_batch, y_batch, z_batch, E_pcb, E_adhesive, E_chip, mu_pcb, mu_adhesive, mu_chip)
        alpha = get_alpha(x_batch, y_batch, z_batch, alpha_pcb, alpha_adhesive, alpha_chip)
        rho, c = get_rho_and_c(x_batch, y_batch, z_batch, rho_pcb, rho_adhesive, rho_chip, c_pcb, c_adhesive, c_chip)

        loss_value = losstem(x_batch, y_batch, z_batch, t_batch, T_actual_batch, Tmodel, k_pcb, k_adhesive, k_chip, rho_pcb,
                             rho_adhesive, rho_chip, c_pcb, c_adhesive, c_chip)
        loss_value += displaceloss(x_batch, y_batch, z_batch, t_batch, model_u, mu, alpha, model)
        loss_value += boundary_loss(x_batch, y_batch, z_batch, t_batch, model_u)
        loss_value += ref_temp_loss(x_batch, y_batch, z_batch, t_batch, model_u, Tmodel)

        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {average_loss:.4f}")

    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch, z_batch, t_batch, T_actual_batch in val_loader:
            x_batch, y_batch, z_batch, t_batch, T_actual_batch = (
                x_batch.to(device),
                y_batch.to(device),
                z_batch.to(device),
                t_batch.to(device),
                T_actual_batch.to(device),
            )
            E, mu = get_E_and_mu(x_batch, y_batch, z_batch, E_pcb, E_adhesive, E_chip, mu_pcb, mu_adhesive, mu_chip)
            alpha = get_alpha(x_batch, y_batch, z_batch, alpha_pcb, alpha_adhesive, alpha_chip)
            rho, c = get_rho_and_c(x_batch, y_batch, z_batch, rho_pcb, rho_adhesive, rho_chip, c_pcb, c_adhesive, c_chip)

            loss_value = losstem(x_batch, y_batch, z_batch, t_batch, T_actual_batch, Tmodel, k_pcb, k_adhesive, k_chip, rho_pcb,
                                 rho_adhesive, rho_chip, c_pcb, c_adhesive, c_chip)
            loss_value += displaceloss(x_batch, y_batch, z_batch, t_batch, model_u, mu, alpha, model)
            loss_value += boundary_loss(x_batch, y_batch, z_batch, t_batch, model_u)
            loss_value += ref_temp_loss(x_batch, y_batch, z_batch, t_batch, model_u, Tmodel)

            val_loss += loss_value.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

# 在训练结束后保存模型
model_save_path = 'path/to/save/model.pth'  # 指定模型保存的路径
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# 将预测结果存储到DataFrame中
results = pd.DataFrame({

    'x': x.cpu().numpy(),
    'y': y.cpu().numpy(),
    'z': z.cpu().numpy(),
    't': t.cpu().numpy(),
    'T': T.cpu().detach().numpy().flatten(),
    'u': u.cpu().detach().numpy().flatten(),
    'v': v.cpu().detach().numpy().flatten(),
    'w': w.cpu().detach().numpy().flatten(),
    'sigma_xx': sigma_xx.cpu().detach().numpy().flatten(),
    'sigma_yy': sigma_yy.cpu().detach().numpy().flatten(),
    'sigma_zz': sigma_zz.cpu().detach().numpy().flatten(),
    'sigma_xy': sigma_xy.cpu().detach().numpy().flatten(),
    'sigma_xz': sigma_xz.cpu().detach().numpy().flatten(),
    'sigma_yz': sigma_yz.cpu().detach().numpy().flatten()
})

# 将结果保存到CSV文件中
results.to_csv('results.csv', index=False)

