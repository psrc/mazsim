import orca


@orca.step('custom_step_example')
def custom_step_example():
    year = orca.get_injectable('year')
    job_count = len(orca.get_table('jobs'))
    hh_count = len(orca.get_table('households'))
    print(f"Year: {year}, Job count: {job_count}, Household count: {hh_count}")