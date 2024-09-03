# change urdf to given finger length and fingertip length
from urdf_parser_py.urdf import URDF
import xml.etree.ElementTree as ET
import os
print (os.getcwd())

def change_parameter_lsc_urdf(tip_length,
                              curve_value,
                              urdf_path='asset/lc_soft_enable_wide_grip/lc_soft_enable_wide_grip.urdf'):
    ## input finger length and tip length
    ## output status of change: filename - success
    
    # urdf_path='lc_soft_enable_wide_grip.urdf'
    robot = URDF.from_xml_file(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    if tip_length==0:
        tip_length=20
    if curve_value==0:
        curve_value=2
 
    filename_tl='asset/lc_soft_enable_wide_grip/meshes/right_distal_'+str(tip_length)+'_'+str(curve_value)+'.stl'
    print ("filename for tip length:",filename_tl)
    if os.path.isfile(filename_tl) == False:
        print ("wrong file - probably length not in range")
        return(0)
    cfilename_tl=filename_tl
    
    for i in range(9):
        if root[i].attrib['name']=='right_tip_link':
            root[i][0][0][0].attrib['filename']=cfilename_tl
            root[i][0][0][0].attrib['scale']="0.001 0.001 0.001"
            root[i][1][0][0].attrib['filename']=filename_tl
            root[i][1][0][0].attrib['scale']="0.001 0.001 0.001"
            
        if root[i].attrib['name']=='left_tip_link':
            root[i][0][0][0].attrib['filename']=cfilename_tl.replace('right','left')
            root[i][0][0][0].attrib['scale']="0.001 0.001 0.001"
            root[i][1][0][0].attrib['filename']=filename_tl.replace('right','left')
            root[i][1][0][0].attrib['scale']="0.001 0.001 0.001"
    
    tree.write(urdf_path)
    return (urdf_path)

if __name__=="__main__":
  change_parameter_lsc_urdf(50,6)
  

# def change_parameter_lsc_urdf(finger_length,
#                               tip_length,
#                               urdf_path='asset/lc_soft_enable_wide_grip/lc_soft_enable_wide_grip.urdf'):
#     ## input finger length and tip length
#     ## output status of change: filename - success
    
#     robot = URDF.from_xml_file(urdf_path)
#     tree = ET.parse(urdf_path)
#     root = tree.getroot()
#     if finger_length==0:
#         finger_length=60
#     filename_fl='asset/lc_soft_enable_wide_grip/meshes/right_base_'+str(finger_length)+'.stl'
#     # print ("filename for finger length:",filename_fl)
#     if os.path.isfile(filename_fl) == False:
#         print ("wrong file - probably length not in range")
#         return(0)    
#     cfilename_fl=filename_fl
#     if tip_length==0:
#         tip_length=20
 
#     filename_tl='asset/lc_soft_enable_wide_grip/meshes/right_distal_'+str(tip_length)+'.stl'
#     # print ("filename for tip length:",filename_tl)
#     if os.path.isfile(filename_tl) == False:
#         print ("wrong file - probably length not in range")
#         return(0)
#     cfilename_tl=filename_tl
    
#     for i in range(9):
#         if root[i].attrib['name']=='right_finger_link':
#             root[i][0][0][0].attrib['filename']=cfilename_fl
#             root[i][1][0][0].attrib['filename']=filename_fl
            
#         if root[i].attrib['name']=='right_tip_link':
#             root[i][0][0][0].attrib['filename']=cfilename_tl
#             root[i][1][0][0].attrib['filename']=filename_tl    
  
#         if root[i].attrib['name']=='left_finger_link':
#             root[i][0][0][0].attrib['filename']=cfilename_fl.replace('right','left')
#             root[i][1][0][0].attrib['filename']=filename_fl.replace('right','left')
            
#         if root[i].attrib['name']=='left_tip_link':
#             root[i][0][0][0].attrib['filename']=cfilename_tl.replace('right','left')
#             root[i][1][0][0].attrib['filename']=filename_tl.replace('right','left')
            
#         if root[i].attrib['name']=='right_tip':
            
#             root[i][4].attrib['xyz']=str((finger_length+9.5)/1000)+' '+str(0.00)+' '+str(0.005375)
#             # print(root[i][4].attrib['xyz'])
#         if root[i].attrib['name']=='left_tip':
            
#             root[i][4].attrib['xyz']=str((finger_length+9.5)/1000)+' '+str(0.00)+' '+str(0.005375)
    
#     tree.write(urdf_path)
#     return (urdf_path)



# if __name__=="__main__":
#   change_parameter_lsc_urdf(60,60)
  

