from json import dump
from sys import argv


def loads(s):
    pred_strs = s[2:-2].replace('), (', ')\n(').split('\n')
    output = []
    for pred in pred_strs:
        output.append([pred[1:-1].replace("', '", "'\n'").split('\n')[0][1:-1]])
    return output

def index_log_avg(f):
    with open(f, 'r') as file:
        s = file.read().split('\n')
    return sum([int(i.split(' ')[-7]) for i in s]) / len(s)


def get_template(s):
    """
    :param s: A string of a filter
    :return: The template that string is from
    """
    filter = loads(s)
    if len(filter) == 1:
        return '1'
    if len(filter) == 5:
        return '6'
    if filter[0][0] == filter[1][0] == 'o_orderdate':
        return '10/4'
    if filter[0][0] == filter[1][0] == 'l_receiptdate':
        return '12'
    if (filter[0][0] == 'l_shipdate') and filter[1][0] == 'o_orderdate':
        return '3'



def approx(n, v):
    return (n // (0.1 ** v)) / (10 ** v)


def logs_to_jsons(input, output_folder, print_things=False):
    """
    Convert raw logs into a more workable json format
    :param input: The location of a file containing logs
    :param output_folder: A folder to dump times.json, rows.json, and total_rows.json
    :param print_things: Whether to print summary statistics
    """
    with open(input, "r") as file:
        logs = file.read()
    times = {}
    rows = {}
    total_rows = {}
    logs = logs.split('Partitions: ')
    for section in logs:
        if section == "":
            continue
        new_section = section.replace('Filters ', '').replace(' using ', '?').replace(' processes scanned ', '?').replace(' out of ', '?').replace(' rows (', '?').replace('%) in ', '?').replace(' seconds.', '')
        sect_list_iter = new_section.split('\n').__iter__()
        folder = sect_list_iter.__next__()
        times[folder] = {}
        rows[folder] = {}
        total_rows[folder] = {}
        for row in sect_list_iter:
            if row == "":
                continue
            qs, processes, scanned_rows, all_rows, percentage, time = row.split('?')
            processes = int(processes)
            scanned_rows = int(scanned_rows)
            all_rows = int(all_rows)
            time = float(time)
            total_rows[folder][qs] = all_rows
            rows[folder][qs] = scanned_rows
            if processes not in times[folder]:
                times[folder][processes] = {}
            times[folder][processes][qs] = time
    with open(output_folder + "/times.json", "w") as file:
        dump(times, file)
    with open(output_folder + "/total_rows.json", "w") as file:
        dump(total_rows, file)
    with open(output_folder + "/rows.json", "w") as file:
        dump(rows, file)
    if print_things:
        print("Folder\t\tTemplate\tRows Scanned\tTotal Rows\tProcesses\tFile Bandwidth\tTrue Bandwidth\tTime")
        for folder in times:
            for processes in times[folder]:
                template_time = {}
                template_scanned_rows = {}
                template_all_rows = {}
                for filter in times[folder][processes].keys():
                    t = get_template(filter)
                    template_time[t] = times[folder][processes][filter] + template_time.get(t, 0)
                    template_all_rows[t] = total_rows[folder][filter] + template_all_rows.get(t, 0)
                    template_scanned_rows[t] = rows[folder][filter] + template_scanned_rows.get(t, 0)
                for t in template_time.keys():
                    t_time = template_time[t]
                    t_all_rows = template_all_rows[t]
                    t_rows = template_scanned_rows[t]
                    f_bandwidth = approx(t_all_rows / t_time, 5)
                    t_bandwidth = approx(t_rows / t_time, 5)
                    print(f"{folder}    \t{t}\t\t{t_rows}  \t{t_all_rows}  \t{processes}\t\t{f_bandwidth}\t{t_bandwidth}\t{t_time}")
                total_time = sum(times[folder][processes].values())
                total_scanned_rows = sum(rows[folder].values())
                total_all_rows = sum(total_rows[folder].values())
                f_bandwidth = approx(total_all_rows / total_time, 5)
                t_bandwidth = approx(total_scanned_rows / total_time, 5)
                print(f"{folder}    \tTotal\t\t{total_scanned_rows}\t{total_all_rows}\t{processes}\t\t{f_bandwidth}\t{t_bandwidth}\t{total_time}")

if __name__ == "__main__":
    logs_to_jsons(argv[1], argv[2], len(argv) > 3)



