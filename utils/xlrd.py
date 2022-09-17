import xlwt
import xlrd

def save_output2d(data, sheet_name, workbook):

    sheet = workbook.add_sheet(sheet_name)

    for i in range(0, len(data)):
        
        for j in range(0, len(data[i])):
            
            sheet.write(i, j, data[i][j])

def save_output1d(data, sheet_name, workbook):
    sheet = workbook.add_sheet(sheet_name)

    for i in range(0, len(data)):        
            sheet.write(i, 0, data[i])